"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import time
import random
import numpy as np
import atexit

import torch
import torch.nn as nn
import torch.distributed
import torch.backends.cudnn

from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data import DistributedSampler
# from torch.utils.data.dataloader import DataLoader
from ..data import DataLoader


def setup_distributed(
    print_rank: int = 0,
    print_method: str = 'builtin',
    seed: int = None,
    deterministic: bool = False,
):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
        deterministic, enable deterministic CUDA/PyTorch algorithms when true,
    """
    try:
        # https://pytorch.org/docs/stable/elastic/run.html
        RANK = int(os.getenv('RANK', -1))
        LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
        WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

        # torch.distributed.init_process_group(backend=backend, init_method='env://')
        torch.distributed.init_process_group(init_method='env://')
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        enabled_dist = True
        if get_rank() == print_rank:
            print('Initialized distributed mode...')

    except Exception:
        enabled_dist = False
        print('Not init distributed mode.')

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed, deterministic=deterministic)

    return enabled_dist


def setup_print(is_main, method='builtin'):
    """This function disables printing when not in master process
    """
    import builtins as __builtin__

    if method == 'builtin':
        builtin_print = __builtin__.print

    elif method == 'rich':
        import rich
        builtin_print = rich.print

    else:
        raise AttributeError('')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@atexit.register
def cleanup():
    """cleanup distributed environment
    """
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)



def warp_model(
    model: torch.nn.Module,
    sync_bn: bool=False,
    dist_mode: str='ddp',
    find_unused_parameters: bool=False,
    compile: bool=False,
    compile_mode: str='reduce-overhead',
    **kwargs
):
    if is_dist_available_and_initialized():
        rank = get_rank()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        if dist_mode == 'dp':
            model = DP(model, device_ids=[rank], output_device=rank)
        elif dist_mode == 'ddp':
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
        else:
            raise AttributeError('')

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model

def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers)
    return loader



def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size

        return {k: v for k, v in zip(keys, values)}


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def sync_time():
    """sync_time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()



def setup_seed(seed: int, deterministic: bool = False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUBLAS_WORKSPACE_CONFIG must be present before the first CUDA
        # matmul. The root launcher sets it before spawning train.py; keep
        # this fallback here as well for direct ``python train.py`` usage.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

        # Disable algorithm/autotuner choices that can vary between runs.
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('highest')

        # TinyFormer's deformable-attention extension contains CUDA backward
        # paths for which PyTorch cannot promise a deterministic kernel on all
        # versions. ``warn_only`` keeps the training job runnable while
        # enabling deterministic kernels everywhere PyTorch supports them.
        # PyTorch emits a warning if it encounters an unsupported operation.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # compatibility with older PyTorch releases
            torch.use_deterministic_algorithms(True)


def capture_rng_state():
    """Capture Python, NumPy, CPU and CUDA RNG states for checkpoint resume."""

    local_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        local_state['cuda'] = torch.cuda.get_rng_state_all()

    # Checkpoints are produced by every rank (``save_on_master`` only filters
    # the final write), so retain one RNG snapshot per rank when distributed.
    if is_dist_available_and_initialized():
        return {
            'version': 1,
            'world_size': get_world_size(),
            'rank_states': all_gather(local_state),
        }
    return {'version': 1, 'world_size': 1, 'rank_states': [local_state]}


def restore_rng_state(state) -> bool:
    """Restore the current rank's RNG state from a checkpoint snapshot."""

    if not isinstance(state, dict):
        return False
    rank_states = state.get('rank_states')
    if isinstance(rank_states, list):
        rank = get_rank()
        if rank >= len(rank_states):
            return False
        state = rank_states[rank]
    if not isinstance(state, dict):
        return False

    try:
        if 'python' in state:
            random.setstate(state['python'])
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
        if 'torch' in state:
            torch.random.set_rng_state(state['torch'])
        if torch.cuda.is_available() and state.get('cuda') is not None:
            torch.cuda.set_rng_state_all(state['cuda'])
    except (TypeError, ValueError, RuntimeError) as error:
        print(f'Warning: could not restore complete RNG state: {error}')
        return False
    return True


# for torch.compile
def check_compile():
    import torch
    import warnings
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    return gpu_ok

def is_compile(model):
    import torch._dynamo
    return type(model) in (torch._dynamo.OptimizedModule, )

def de_complie(model):
    return model._orig_mod if is_compile(model) else model

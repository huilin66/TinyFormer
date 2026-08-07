"""Structure tests for arbitrary-modality TinyFormer fusion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.multimodal import (  # noqa: E402
    AddFusion,
    ConcatFusion,
    DetectionQueryFusion,
    FUSION_MODES,
    MultiModalTinyFormer,
    SSA4ScaleStage,
)


class DummyBackbone(nn.Module):
    def forward(self, image):
        return image, image * 2, image * 3


class DummySSA(nn.Module):
    def forward(self, backbone_features, image):
        bias = image.mean(dim=1, keepdim=True)
        return tuple(feature + bias for feature in backbone_features)


class DummyNeck(nn.Module):
    def forward(self, features):
        return tuple(feature + 1 for feature in features)


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = 3
        self.input_proj = nn.ModuleList([nn.Identity() for _ in range(self.num_levels)])

    def _get_encoder_input(self, features):
        flattened = []
        shapes = []
        for feature in features:
            height, width = feature.shape[-2:]
            flattened.append(feature.flatten(2).permute(0, 2, 1))
            shapes.append([height, width])
        return torch.cat(flattened, dim=1), shapes

    def forward_from_memory(self, memory, spatial_shapes, targets=None):
        del spatial_shapes, targets
        pooled = memory.mean(dim=1)
        return {
            "pred_logits": pooled.unsqueeze(1),
            "pred_boxes": pooled[:, :1].unsqueeze(1).repeat(1, 1, 4),
        }

    def forward(self, features, targets=None):
        memory, spatial_shapes = self._get_encoder_input(features)
        return self.forward_from_memory(memory, spatial_shapes, targets)


def build_model(mode, *, num_modalities=3, share_weight=False, channels=None):
    return MultiModalTinyFormer(
        backbone=DummyBackbone(),
        ssa=DummySSA(),
        neck=DummyNeck(),
        decoder=DummyDecoder(),
        fusion=AddFusion(normalize=True),
        image_fusion=ConcatFusion(dim=1),
        final_fusion=DetectionQueryFusion(),
        fusion_mode=mode,
        num_modalities=num_modalities,
        modality_channels=channels or [3] * num_modalities,
        share_weight=share_weight,
    )


class FusionTests(unittest.TestCase):
    def test_add_and_concat_support_arbitrary_n_and_feature_pyramids(self):
        features = [
            (torch.full((2, 4, 8, 8), float(index)), torch.full((2, 4, 4, 4), float(index)))
            for index in range(1, 5)
        ]
        added = AddFusion(normalize=True)(features)
        concatenated = ConcatFusion()(features)
        self.assertEqual(added[0].shape, (2, 4, 8, 8))
        self.assertTrue(torch.allclose(added[0], torch.full_like(added[0], 2.5)))
        self.assertEqual(concatenated[0].shape, (2, 16, 8, 8))

    def test_recursive_prediction_dictionary_fusion(self):
        outputs = [
            {"pred_logits": torch.full((1, 2, 3), value), "aux": [{"x": torch.full((1, 1), value)}]}
            for value in (1.0, 3.0, 5.0)
        ]
        fused = AddFusion(normalize=True)(outputs)
        self.assertTrue(torch.allclose(fused["pred_logits"], torch.full((1, 2, 3), 3.0)))
        self.assertEqual(fused["aux"][0]["x"].item(), 3.0)

    def test_recursive_fusion_preserves_integer_decoder_metadata(self):
        positive_idx = (torch.tensor([0, 2], dtype=torch.long),)
        outputs = [
            {
                "pred_logits": torch.full((1, 2, 3), value),
                "dn_meta": {
                    "dn_positive_idx": tuple(index.clone() for index in positive_idx),
                    "dn_num_group": 2,
                    "dn_num_split": [4, 2],
                },
            }
            for value in (1.0, 3.0)
        ]
        fused = AddFusion(normalize=True)(outputs)
        fused_index = fused["dn_meta"]["dn_positive_idx"][0]
        self.assertEqual(fused_index.dtype, torch.long)
        self.assertTrue(torch.equal(fused_index, positive_idx[0]))
        self.assertTrue(torch.allclose(fused["pred_logits"], torch.full((1, 2, 3), 2.0)))

    def test_recursive_fusion_rejects_mismatched_integer_metadata(self):
        outputs = [
            {"pred_logits": torch.ones(1), "index": torch.tensor([0], dtype=torch.long)},
            {"pred_logits": torch.ones(1), "index": torch.tensor([1], dtype=torch.long)},
        ]
        with self.assertRaisesRegex(ValueError, "integer/bool tensor metadata"):
            AddFusion(normalize=True)(outputs)

    def test_final_fusion_concatenates_unaligned_queries(self):
        outputs = [
            {
                "pred_logits": torch.full((1, 2, 3), value),
                "pred_boxes": torch.full((1, 2, 4), value),
            }
            for value in (1.0, 3.0)
        ]
        fused = DetectionQueryFusion()(outputs)
        self.assertEqual(fused["pred_logits"].shape, (1, 4, 3))
        self.assertTrue(torch.equal(fused["pred_logits"][:, :2], outputs[0]["pred_logits"]))
        self.assertTrue(torch.equal(fused["pred_logits"][:, 2:], outputs[1]["pred_logits"]))

    def test_final_fusion_offsets_denoising_indices(self):
        outputs = []
        for value in (1.0, 2.0):
            outputs.append(
                {
                    "pred_logits": torch.full((1, 2, 3), value),
                    "dn_outputs": [{"pred_logits": torch.full((1, 4, 3), value)}],
                    "dn_meta": {
                        "dn_positive_idx": (torch.tensor([0, 2]),),
                        "dn_num_group": 2,
                        "dn_num_split": [4, 2],
                    },
                }
            )
        fused = DetectionQueryFusion()(outputs)
        self.assertEqual(fused["dn_outputs"][0]["pred_logits"].shape[1], 8)
        self.assertTrue(
            torch.equal(fused["dn_meta"]["dn_positive_idx"][0], torch.tensor([0, 2, 4, 6]))
        )
        self.assertEqual(fused["dn_meta"]["dn_num_group"], 4)
        self.assertEqual(fused["dn_meta"]["dn_num_split"], [8, 4])


class MultiModalTinyFormerTests(unittest.TestCase):
    def test_all_seven_modes_support_three_modalities(self):
        inputs = torch.randn(2, 9, 16, 16)
        for mode in FUSION_MODES:
            with self.subTest(mode=mode):
                model = build_model(mode)
                self.assertEqual(model.input_channels, 9)
                output = model(inputs)
                expected_queries = 3 if mode == "FF" else 1
                self.assertEqual(output["pred_logits"].shape, (2, expected_queries, 3))
                self.assertEqual(output["pred_boxes"].shape, (2, expected_queries, 4))

    def test_df_fuses_projected_memory_before_one_prediction_head(self):
        model = build_model("DF", num_modalities=2, share_weight=False)
        self.assertEqual(model.decoders.count, 1)
        self.assertEqual(len(model.decoder_projectors), 1)
        calls = []
        handle = model.decoders.at(0).register_forward_hook(lambda *args: calls.append("forward"))
        try:
            output = model(torch.randn(2, 6, 16, 16))
        finally:
            handle.remove()
        self.assertEqual(calls, [])
        self.assertEqual(output["pred_boxes"].shape, (2, 1, 4))

    def test_df_tuning_weights_initialize_every_input_projection(self):
        model = build_model("DF", num_modalities=3, share_weight=False)
        legacy = {"decoder.input_proj.0.weight": torch.ones(1)}
        remapped = model.remap_tuning_state_dict(legacy)
        self.assertIn("decoders.shared.input_proj.0.weight", remapped)
        self.assertIn("decoder_projectors.0.input_proj.0.weight", remapped)
        self.assertIn("decoder_projectors.1.input_proj.0.weight", remapped)

    def test_n_equals_one_is_supported_by_every_mode(self):
        inputs = torch.randn(2, 3, 16, 16)
        for mode in FUSION_MODES:
            with self.subTest(mode=mode):
                output = build_model(mode, num_modalities=1)(inputs)
                self.assertEqual(output["pred_boxes"].shape, (2, 1, 4))

    def test_share_weight_reuses_modules_and_independent_mode_copies_them(self):
        shared = build_model("EF", share_weight=True)
        independent = build_model("EF", share_weight=False)
        self.assertIs(shared.backbones.at(0), shared.backbones.at(2))
        self.assertIs(shared.ssas.at(0), shared.ssas.at(1))
        self.assertIsNot(independent.backbones.at(0), independent.backbones.at(2))
        self.assertIsNot(independent.ssas.at(0), independent.ssas.at(1))

    def test_shared_weights_reject_incompatible_modality_channels(self):
        with self.assertRaisesRegex(ValueError, "identical modality input channels"):
            build_model("EF", num_modalities=2, share_weight=True, channels=[3, 1])

    def test_image_fusion_allows_mixed_channels_without_prefusion_branches(self):
        model = build_model("IF", num_modalities=2, share_weight=True, channels=[3, 1])
        output = model([torch.randn(1, 3, 16, 16), torch.randn(1, 1, 16, 16)])
        self.assertEqual(output["pred_boxes"].shape, (1, 1, 4))

    def test_sequence_and_named_mapping_inputs(self):
        images = [torch.randn(1, 3, 8, 8) for _ in range(3)]
        model = build_model("NF")
        sequence_output = model(images)
        model.modality_names = ["rgb", "infrared", "depth"]
        mapping_output = model(dict(zip(model.modality_names, images)))
        self.assertEqual(sequence_output["pred_logits"].shape, mapping_output["pred_logits"].shape)

    def test_legacy_weights_expand_to_every_independent_branch(self):
        model = build_model("FF", share_weight=False)
        legacy = {
            "backbone.dinov3.weight": torch.ones(1),
            "backbone.sda.weight": torch.ones(1),
            "encoder.weight": torch.ones(1),
            "decoder.weight": torch.ones(1),
        }
        remapped = model.remap_tuning_state_dict(legacy)
        self.assertEqual(sum(key.endswith("dinov3.weight") for key in remapped), 3)
        self.assertEqual(sum(key.endswith("sda.weight") for key in remapped), 3)
        self.assertEqual(sum(key.endswith("necks.branches.0.weight") for key in remapped), 1)
        self.assertEqual(sum(key.endswith("decoders.branches.2.weight") for key in remapped), 1)

    def test_decoupled_ssa_emits_four_expected_scales(self):
        stage = SSA4ScaleStage(embed_dim=8, hidden_dim=4, conv_inplane=2)
        image = torch.randn(2, 3, 64, 64)
        backbone_features = [torch.randn(2, 8, 4, 4) for _ in range(3)]
        outputs = stage(backbone_features, image)
        self.assertEqual([tuple(output.shape) for output in outputs], [
            (2, 4, 16, 16),
            (2, 4, 8, 8),
            (2, 4, 4, 4),
            (2, 4, 2, 2),
        ])


if __name__ == "__main__":
    unittest.main()

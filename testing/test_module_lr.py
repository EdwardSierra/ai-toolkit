"""Unit tests for the per-module lr weights extension (v7 "surgical grail").

Purpose: lock down the fork change that adds kohya-style per-module learning
rate weights (attn / mlp.down / mlp.gate / mlp.up / txtfusion) on top of the
existing per-block weights. Covers:
  1. parse_module_lr_weights: "kind:weight" strings, dict passthrough,
     None/empty -> None, malformed entry raises.
  2. get_module_kind: flat DiT names ($$ peft / _ legacy / . dotted) for
     attn vs mlp submodules, txtfusion layerwise blocks, legacy SD-UNet
     names, and non-matching names -> "other".
  3. Effective per-LoRA lr = base_lr * block_weight * module_weight for the
     v7 config's exact weights, across all 28 blocks, all module kinds, and
     the no-module-weights (block-only) regression path.

Inputs: none (pure CPU). Output: standard unittest results.
Run: cd ai-toolkit && venv/bin/python -m unittest testing.test_module_lr -v
"""
import unittest

from toolkit.kohya_lora import (
    LoRANetwork,
    get_module_kind,
    parse_module_lr_weights,
)


class _Lora:
    def __init__(self, name):
        self.lora_name = name


class _Weights:
    """Minimal stand-in for the network's lr-weight attributes."""

    def __init__(self, down=None, mid=None, up=None, modules=None):
        self.down_lr_weight = down
        self.mid_lr_weight = mid
        self.up_lr_weight = up
        self.module_lr_weights = modules


def _effective_lr(down, mid, up, modules, lora_name):
    w = _Weights(down, mid, up, modules)
    lora = _Lora(lora_name)
    return LoRANetwork.get_lr_weight(w, lora) * LoRANetwork.get_module_lr_weight(w, lora)


class ParseModuleLrWeightsTest(unittest.TestCase):
    def test_string_form(self):
        self.assertEqual(
            parse_module_lr_weights("attn:0.5,mlp.gate:2.5,mlp.up:1.0,mlp.down:0.6"),
            {"attn": 0.5, "mlp.gate": 2.5, "mlp.up": 1.0, "mlp.down": 0.6},
        )

    def test_dict_form_and_whitespace(self):
        self.assertEqual(parse_module_lr_weights({"attn": 0.5}), {"attn": 0.5})
        self.assertEqual(
            parse_module_lr_weights(" attn : 0.5 , mlp.up : 1.0 "),
            {"attn": 0.5, "mlp.up": 1.0},
        )

    def test_none_and_empty(self):
        self.assertIsNone(parse_module_lr_weights(None))
        self.assertIsNone(parse_module_lr_weights(""))
        self.assertIsNone(parse_module_lr_weights(" , "))

    def test_malformed_entry_raises(self):
        with self.assertRaises(ValueError):
            parse_module_lr_weights("mlp.gate")


class GetModuleKindTest(unittest.TestCase):
    def test_flat_peft_names(self):
        self.assertEqual(get_module_kind("transformer$$diffusion_model$$blocks$$18$$attn$$wq"), "attn")
        self.assertEqual(get_module_kind("transformer$$diffusion_model$$blocks$$18$$attn$$gate"), "attn")
        self.assertEqual(get_module_kind("transformer$$diffusion_model$$blocks$$18$$mlp$$gate"), "mlp.gate")
        self.assertEqual(get_module_kind("transformer$$diffusion_model$$blocks$$18$$mlp$$up"), "mlp.up")
        self.assertEqual(get_module_kind("transformer$$diffusion_model$$blocks$$18$$mlp$$down"), "mlp.down")

    def test_flat_legacy_underscore_names(self):
        self.assertEqual(get_module_kind("lora_unet_diffusion_model_blocks_18_mlp_gate"), "mlp.gate")
        self.assertEqual(get_module_kind("lora_unet_diffusion_model_blocks_0_attn_wq"), "attn")

    def test_flat_dotted_saved_keys(self):
        # the exact key format stored in the saved .safetensors files
        self.assertEqual(get_module_kind("diffusion_model.blocks.18.attn.gate"), "attn")
        self.assertEqual(get_module_kind("diffusion_model.blocks.18.mlp.down"), "mlp.down")
        self.assertEqual(get_module_kind("diffusion_model.blocks.0.mlp.up"), "mlp.up")

    def test_txtfusion_layerwise_blocks(self):
        self.assertEqual(
            get_module_kind("transformer$$diffusion_model$$txtfusion$$layerwise_blocks$$0$$mlp$$down"),
            "txtfusion",
        )
        self.assertEqual(
            get_module_kind("diffusion_model.txtfusion.layerwise_blocks.0.attn.wq"),
            "txtfusion",
        )

    def test_legacy_sd_unet_names(self):
        self.assertEqual(
            get_module_kind("lora_unet_down_blocks_0_resnets_0_attention_1_transformer_blocks_0_attn1_to_q"),
            "attn",
        )
        self.assertEqual(get_module_kind("lora_unet_down_blocks_0_resnets_0_ff_net_0"), "mlp")
        self.assertEqual(get_module_kind("lora_unet_down_blocks_0_resnets_0_2"), "other")

    def test_non_matching_names(self):
        self.assertEqual(get_module_kind("lora_te_text_model_encoder_layers_0_mlp_fc1"), "other")
        self.assertEqual(get_module_kind("lora_unet_conv_in"), "other")


class V7EffectiveLrTest(unittest.TestCase):
    """v7 config's exact weights -> expected effective lr multiplier.

    Effective lr = base_lr * block_weight * module_weight, where the block
    mapping for flat DiT naming is: down_lr_weight[0..11] -> blocks 0-11,
    mid_lr_weight -> block 12, up_lr_weight[0..11] -> blocks 13-24,
    blocks 25-27 clamp to the last up entry.
    """

    DOWN = [1.0] * 12
    MID = 1.0
    UP = [1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    MODULES = {"attn": 0.5, "mlp.gate": 2.5, "mlp.up": 1.0, "mlp.down": 0.6}

    def _name(self, block, module, sub):
        return "transformer$$diffusion_model$$blocks$$%d$$%s$$%s" % (block, module, sub)

    def test_front_blocks_module_weights_only(self):
        # block weight 1.0 -> effective == module weight
        self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(0, "attn", "wq")), 0.5)
        self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(0, "mlp", "gate")), 2.5)
        self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(0, "mlp", "down")), 0.6)
        self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(11, "mlp", "up")), 1.0)

    def test_mid_block_module_weights_only(self):
        self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(12, "mlp", "down")), 0.6)
        self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(12, "attn", "wv")), 0.5)

    def test_hot_zone_block_times_module(self):
        # blocks 18-27: block 1.5 * module weight
        for b in (18, 20, 24):
            self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(b, "mlp", "gate")), 3.75)
            self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(b, "attn", "wq")), 0.75)
            self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(b, "mlp", "up")), 1.5)
            self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(b, "mlp", "down")), 0.9)

    def test_final_blocks_clamp_to_last_up_entry(self):
        for b in (25, 26, 27):
            self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, self._name(b, "mlp", "gate")), 3.75)

    def test_unlisted_module_kind_is_neutral(self):
        # txtfusion (not listed in v7) and text-encoder-ish "other" names
        # keep only the block weight.
        tf = "transformer$$diffusion_model$$txtfusion$$layerwise_blocks$$0$$mlp$$down"
        self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, tf), self.DOWN[0])
        self.assertAlmostEqual(
            _effective_lr(self.DOWN, self.MID, self.UP, self.MODULES, "lora_te_text_model_encoder_layers_0_mlp_fc1"), 1.0
        )

    def test_block_only_regression_unchanged(self):
        # no module weights -> identical multipliers to the v6 block-only path
        for b in (0, 12, 14, 18, 27):
            block_w = self.DOWN[b] if b < 12 else (self.MID if b == 12 else self.UP[min(b - 13, len(self.UP) - 1)])
            self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, None, self._name(b, "mlp", "gate")), block_w)
            self.assertAlmostEqual(_effective_lr(self.DOWN, self.MID, self.UP, None, self._name(b, "attn", "wq")), block_w)

    def test_module_only_without_block_lr(self):
        # block weights unset (None) -> effective == module weight for every block
        for b in (0, 12, 18, 27):
            self.assertAlmostEqual(_effective_lr(None, None, None, self.MODULES, self._name(b, "mlp", "gate")), 2.5)
            self.assertAlmostEqual(_effective_lr(None, None, None, self.MODULES, self._name(b, "attn", "wo")), 0.5)


if __name__ == "__main__":
    unittest.main()

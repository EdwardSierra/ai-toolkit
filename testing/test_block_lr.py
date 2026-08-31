"""Unit tests for the per-block lr patch (flat DiT block naming).

Purpose: lock down the fork change that enables kohya-style per-block
learning-rate weights for flat transformer block naming (krea2
"blocks.N"), which previously only worked for SD-UNet
"down_blocks_N_resnets_M" names. Covers:
  1. get_block_index: flat ($$ peft / _ legacy / . dotted) and classic names,
     plus the names that must NOT match (double/single/transformer_blocks).
  2. parse_block_lr_kwargs: comma-list strings, scalar mid, passthrough.
  3. The 28-block krea2 mapping: down[0..11] -> blocks 0-11, mid -> 12,
     up[0..11] -> blocks 13-24, blocks 25-27 clamped to the last up entry.

Inputs: none (pure CPU). Output: standard unittest results.
Run: cd ai-toolkit && venv/bin/python -m pytest testing/test_block_lr.py -v
"""
import unittest

from toolkit.kohya_lora import (
    LoRANetwork,
    get_block_index,
    parse_block_lr_kwargs,
)


class _Lora:
    def __init__(self, name):
        self.lora_name = name


class _Weights:
    """Minimal stand-in for the network's block-lr attributes."""

    def __init__(self, down, mid, up):
        self.down_lr_weight = down
        self.mid_lr_weight = mid
        self.up_lr_weight = up


def _lr_weight(down, mid, up, lora_name):
    return LoRANetwork.get_lr_weight(_Weights(down, mid, up), _Lora(lora_name))


class GetBlockIndexTest(unittest.TestCase):
    def test_flat_peft_names(self):
        # peft_format lora_names use $$ as separator
        self.assertEqual(get_block_index("transformer$$diffusion_model$$blocks$$0$$attn$$wq"), 0)
        self.assertEqual(get_block_index("transformer$$diffusion_model$$blocks$$18$$mlp$$gate"), 18)
        self.assertEqual(get_block_index("transformer$$diffusion_model$$blocks$$27$$mlp$$down"), 27)

    def test_flat_legacy_underscore_names(self):
        # non-peft lora_names use _ as separator
        self.assertEqual(get_block_index("lora_unet_diffusion_model_blocks_0_attn_wq"), 0)
        self.assertEqual(get_block_index("lora_unet_diffusion_model_blocks_18_mlp_gate"), 18)

    def test_flat_dotted_names(self):
        # clean_name style (dotted)
        self.assertEqual(get_block_index("diffusion_model.blocks.18.mlp.gate"), 18)

    def test_classic_sd_unet_names_unchanged(self):
        # down: 1 + idx ; up: NUM_OF_BLOCKS + 1 + idx ; mid: NUM_OF_BLOCKS
        self.assertEqual(get_block_index("lora_unet_down_blocks_0_resnets_0_1"), 1)
        self.assertEqual(get_block_index("lora_unet_up_blocks_0_resnets_0_0"), LoRANetwork.NUM_OF_BLOCKS + 1)
        self.assertEqual(get_block_index("lora_unet_mid_block_0"), LoRANetwork.NUM_OF_BLOCKS)

    def test_non_matching_names(self):
        for name in (
            "lora_unet_double_blocks_3_attn1",
            "transformer$$transformer_blocks$$3$$attn$$wq",
            "lora_unet_single_blocks_5_attn",
            "lora_te_text_model_encoder_layers_0_mlp_fc1",
            "lora_unet_conv_in",
        ):
            self.assertEqual(get_block_index(name), -1, name)


class ParseBlockLrKwargsTest(unittest.TestCase):
    def test_comma_lists_and_scalar_mid(self):
        d, m, u = parse_block_lr_kwargs(
            {
                "down_lr_weight": "0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45",
                "mid_lr_weight": 1.0,
                "up_lr_weight": "1.0,1.0,1.0,1.0,1.0,2.5,2.5,2.5,2.5,2.5,2.5,2.5",
            }
        )
        self.assertEqual(d, [0.45] * 12)
        self.assertEqual(m, 1.0)
        self.assertEqual(u, [1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])

    def test_unset_returns_none_triple(self):
        d, m, u = parse_block_lr_kwargs({})
        self.assertIsNone(d)
        self.assertIsNone(m)
        self.assertIsNone(u)


class Krea2BlockMappingTest(unittest.TestCase):
    """The v6 config's exact weights -> expected per-block lr multiplier."""

    DOWN = [0.45] * 12
    MID = 1.0
    UP = [1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]

    def _name(self, block):
        return f"transformer$$diffusion_model$$blocks$$ {block}$$mlp$$gate".replace("$$ ", "$$")

    def test_front_blocks_train_slower(self):
        for b in range(0, 12):
            self.assertAlmostEqual(_lr_weight(self.DOWN, self.MID, self.UP, self._name(b)), 0.45)

    def test_mid_block_at_base_rate(self):
        self.assertAlmostEqual(_lr_weight(self.DOWN, self.MID, self.UP, self._name(12)), 1.0)

    def test_mid_to_late_blocks_at_base_rate(self):
        for b in range(13, 18):
            self.assertAlmostEqual(_lr_weight(self.DOWN, self.MID, self.UP, self._name(b)), 1.0)

    def test_hot_zone_blocks_train_faster(self):
        for b in range(18, 25):
            self.assertAlmostEqual(_lr_weight(self.DOWN, self.MID, self.UP, self._name(b)), 2.5)

    def test_final_blocks_clamp_to_last_up_entry(self):
        for b in range(25, 28):
            self.assertAlmostEqual(_lr_weight(self.DOWN, self.MID, self.UP, self._name(b)), 2.5)

    def test_unset_weights_are_neutral(self):
        self.assertEqual(_lr_weight(None, None, None, self._name(18)), 1.0)


if __name__ == "__main__":
    unittest.main()

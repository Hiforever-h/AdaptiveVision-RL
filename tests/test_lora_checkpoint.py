import unittest

from adaptive_vision_rl.verl.lora_checkpoint import select_lora_tensors


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self


class LoraCheckpointTests(unittest.TestCase):
    def test_selects_paired_lora_weights_with_peft_export_names(self):
        state = {
            "base_model.model.language_model.layers.0.q_proj.lora_A.default.weight": FakeTensor((64, 2560)),
            "base_model.model.language_model.layers.0.q_proj.lora_B.default.weight": FakeTensor((2560, 64)),
            "base_model.model.language_model.layers.0.q_proj.base_layer.weight": FakeTensor((2560, 2560)),
        }
        result = select_lora_tensors(state, rank=64)
        self.assertEqual(
            set(result),
            {
                "base_model.model.language_model.layers.0.q_proj.lora_A.weight",
                "base_model.model.language_model.layers.0.q_proj.lora_B.weight",
            },
        )

    def test_rejects_empty_or_unpaired_lora_state(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_lora_tensors({}, rank=64)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_lora_tensors(
                {"layer.q_proj.lora_A.default.weight": FakeTensor((64, 2560))},
                rank=64,
            )

    def test_rejects_wrong_rank(self):
        state = {
            "layer.q_proj.lora_A.default.weight": FakeTensor((32, 2560)),
            "layer.q_proj.lora_B.default.weight": FakeTensor((2560, 64)),
        }
        with self.assertRaisesRegex(ValueError, "Unexpected LoRA A shape"):
            select_lora_tensors(state, rank=64)

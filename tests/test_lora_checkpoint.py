import unittest

from adaptive_vision_rl.verl.lora_checkpoint import collect_checkpoint_lora_params


class FakeTensor:
    def contiguous(self):
        return self


class LoraCheckpointTests(unittest.TestCase):
    def test_empty_layered_result_falls_back_to_full_fsdp(self):
        module = object()
        tensor = FakeTensor()
        calls = []

        def layered(value):
            self.assertIs(value, module)
            calls.append("layered")
            return {}

        def full(value):
            self.assertIs(value, module)
            calls.append("full")
            return {"lora_A.weight": tensor}

        result = collect_checkpoint_lora_params(
            module, layered_collect=layered, full_collect=full
        )
        self.assertEqual(calls, ["layered", "full"])
        self.assertIs(result["lora_A.weight"], tensor)

    def test_zero_tensors_after_fallback_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "zero tensors"):
            collect_checkpoint_lora_params(
                object(), layered_collect=lambda _: {}, full_collect=lambda _: {}
            )

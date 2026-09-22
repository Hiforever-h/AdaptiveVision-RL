import unittest
from dataclasses import dataclass, field

from adaptive_vision_rl.vllm_compat import _patch_qwen3vl_mapping


@dataclass
class FakeMultiModelKeys:
    language_model: list[str] = field(default_factory=list)
    connector: list[str] = field(default_factory=list)
    tower_model: list[str] = field(default_factory=list)

    @classmethod
    def from_string_field(cls, language_model, connector, tower_model):
        return cls(
            language_model=[language_model],
            connector=[connector],
            tower_model=[tower_model],
        )


class VllmCompatTests(unittest.TestCase):
    def test_backports_broken_qwen3vl_mapping(self):
        class Model:
            def get_mm_mapping(self):
                return FakeMultiModelKeys.from_string_field(
                    language_model="language_model",
                    connector="model.visual.merger",
                    tower_model="model.visual.",
                )

        self.assertEqual(
            _patch_qwen3vl_mapping(Model, FakeMultiModelKeys),
            "patched",
        )
        mapping = Model.get_mm_mapping(None)
        self.assertEqual(mapping.language_model, ["language_model"])
        self.assertEqual(mapping.connector, ["visual.merger"])
        self.assertEqual(mapping.tower_model, ["visual."])

    def test_does_not_replace_an_already_fixed_mapping(self):
        class Model:
            def get_mm_mapping(self):
                return FakeMultiModelKeys.from_string_field(
                    language_model="language_model",
                    connector="visual.merger",
                    tower_model="visual.",
                )

        original = Model.get_mm_mapping
        self.assertEqual(
            _patch_qwen3vl_mapping(Model, FakeMultiModelKeys),
            "already_fixed",
        )
        self.assertIs(Model.get_mm_mapping, original)

    def test_refuses_an_unknown_vendor_mapping(self):
        class Model:
            def get_mm_mapping(self):
                return FakeMultiModelKeys.from_string_field(
                    language_model="language_model",
                    connector="vendor.connector",
                    tower_model="vendor.visual.",
                )

        original = Model.get_mm_mapping
        self.assertEqual(
            _patch_qwen3vl_mapping(Model, FakeMultiModelKeys),
            "unexpected_mapping",
        )
        self.assertIs(Model.get_mm_mapping, original)


if __name__ == "__main__":
    unittest.main()

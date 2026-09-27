"""Check the temporary merged-model evaluation orchestration without a GPU."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sft import evaluate


class SftEvaluateBackendTests(unittest.TestCase):
    def test_base_model_flag_reaches_child_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "evaluate_dtpo.py").write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "assert '--base-model' in sys.argv\n"
                "assert '--checkpoint' not in sys.argv\n"
                "out = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'summary.json').write_text(json.dumps({"
                "'sample_count': 300, 'split': 'val', 'base_model': True, "
                "'adapter_path': None}))\n"
            )
            with patch.object(evaluate, "ROOT", root):
                summary = evaluate.run_evaluation(
                    adapter=None, split="val", output_dir=root / "results",
                    dataset_root=root / "data", model="base", overwrite=False,
                    batch_size=1,
                )
            self.assertTrue(summary["base_model"])

    def test_base_model_mode_needs_no_adapter_and_runs_val_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            config = root / "config.json"
            config.write_text(json.dumps({
                "model": str(model), "data_dir": str(root / "data"),
                "output_dir": str(root / "missing_checkpoints"),
            }))
            output = root / "results"
            calls = []

            def fake_evaluate(**kwargs):
                calls.append(kwargs)
                return {"metrics": {"overall": {"accuracy": 0.5}}}

            argv = ["sft.evaluate", "--config", str(config), "--base-model",
                    "--output-dir", str(output), "--test"]
            with patch.object(sys, "argv", argv), \
                 patch.object(evaluate, "run_evaluation", side_effect=fake_evaluate):
                evaluate.main()
            self.assertEqual([call["split"] for call in calls], ["val", "test"])
            self.assertTrue(all(call["adapter"] is None for call in calls))
            self.assertEqual(json.loads((output / "base_model.json").read_text())["test_accuracy"], 0.5)

    def test_temporary_merged_model_is_used_and_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seen = {}

            def fake_merge(command, **kwargs):
                self.assertEqual(command[1:3], ["-m", "sft.merge_lora"])
                merged = Path(command[command.index("--output-dir") + 1])
                merged.mkdir()
                (merged / "config.json").write_text("{}")
                seen["merged"] = merged

            def fake_evaluate(**kwargs):
                self.assertEqual(kwargs["merged_model"], seen["merged"])
                self.assertTrue(seen["merged"].is_dir())
                return {"sample_count": 300}

            with patch.object(evaluate.subprocess, "run", side_effect=fake_merge), \
                 patch.object(evaluate, "run_evaluation", side_effect=fake_evaluate):
                result = evaluate.evaluate_adapter(
                    adapter=root / "checkpoint-70", split="val",
                    output_dir=root / "val_half", dataset_root=root / "data",
                    model="Qwen/Qwen3-VL-4B-Thinking", overwrite=False,
                    batch_size=16, dynamic_lora=False, temporary_root=root,
                )
            self.assertEqual(result["sample_count"], 300)
            self.assertFalse(seen["merged"].exists())

    def test_merged_flag_and_child_error_are_preserved_in_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "evaluate_dtpo.py").write_text(
                "import sys\n"
                "print('ROOT_CAUSE: simulated failure', flush=True)\n"
                "print('--merged-model' in sys.argv, flush=True)\n"
                "sys.exit(3)\n"
            )
            with patch.object(evaluate, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "evaluator.log"):
                    evaluate.run_evaluation(
                        adapter=root / "adapter", split="val", output_dir=root / "out",
                        dataset_root=root / "data", model="base", overwrite=False,
                        batch_size=1, merged_model=root / "merged",
                    )
            log = (root / "out" / "evaluator.log").read_text()
            self.assertIn("ROOT_CAUSE: simulated failure", log)
            self.assertIn("True", log)


if __name__ == "__main__":
    unittest.main()

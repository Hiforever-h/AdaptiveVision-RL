import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.probe_thinking_format import (
    Generation,
    ProbeSample,
    load_val_samples,
    run_probe,
    summarize,
)


class ThinkingFormatProbeTests(unittest.TestCase):
    def test_parquet_loader_uses_env_kwargs_not_placeholder_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "val/images").mkdir(parents=True)
            (root / "val/lowres").mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(root / "val/images/a.png")
            Image.new("RGB", (2, 2)).save(root / "val/lowres/a.png")
            row = {
                "prompt": [{"role": "user", "content": "Placeholder, not the question"}],
                "env_kwargs": {
                    "sample_id": "a",
                    "split": "val",
                    "question": "Actual question?",
                    "image_path": "val/images/a.png",
                    "lowres_path": "val/lowres/a.png",
                    "reference_boxes": [[0.0, 0.0, 0.5, 0.5]],
                    "source_use_tool": True,
                },
            }
            parquet = root / "val.parquet"
            pq.write_table(pa.Table.from_pylist([row]), parquet)
            samples = load_val_samples(parquet, root, offset=0, limit=1)
            self.assertEqual(samples[0].question, "Actual question?")
            self.assertEqual(samples[0].reference_box, (0.0, 0.0, 0.5, 0.5))

    def test_probe_checks_natural_and_reference_crop_second_turns(self):
        class FakeGenerator:
            def generate(self, texts, images):
                answers = []
                for text, group in zip(texts, images, strict=True):
                    if len(group) == 1:
                        response = (
                            "<answer>1</answer>"
                            if "Question: A?" in text
                            else '<think>Need detail.</think><tool_call>'
                            '{"name":"request_local_region",'
                            '"arguments":{"bbox_2d":[0,0,500,500]}}</tool_call>'
                        )
                    else:
                        response = (
                            "<think>The reference crop shows 1.</think><answer>1</answer>"
                            if "Question: A?" in text
                            else "<think>The requested crop shows 2.</think><answer>2</answer>"
                        )
                    answers.append(Generation(response, "stop", 12))
                return answers

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for sample_id in ("a", "b"):
                full = root / f"{sample_id}-full.png"
                low = root / f"{sample_id}-low.png"
                Image.new("RGB", (4, 4)).save(full)
                Image.new("RGB", (2, 2)).save(low)
                samples.append(
                    ProbeSample(
                        sample_id=sample_id,
                        question=f"{sample_id.upper()}?",
                        lowres_path=low,
                        image_path=full,
                        reference_box=(0.0, 0.0, 0.5, 0.5),
                        source_use_tool=sample_id == "b",
                    )
                )
            records = run_probe(
                FakeGenerator(), samples, batch_size=2, reference_second_probes=1
            )

        self.assertFalse(records[0]["first_turn"]["valid_action"])
        self.assertIsNotNone(records[0]["reference_second_turn"])
        self.assertIsNone(records[0]["natural_second_turn"])
        self.assertTrue(records[1]["first_turn"]["valid_action"])
        self.assertIsNotNone(records[1]["natural_second_turn"])
        self.assertIsNone(records[1]["reference_second_turn"])
        metrics = summarize(records)
        self.assertEqual(metrics["first_turn"]["valid_action_rate"], 0.5)
        self.assertEqual(metrics["natural_second_turn"]["valid_action_rate"], 1.0)
        self.assertEqual(metrics["reference_second_turn"]["valid_action_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

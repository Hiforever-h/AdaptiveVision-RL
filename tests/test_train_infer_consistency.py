"""Exercise batch evaluation through the real prompt/image preparation path."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageOps

from adaptive_vision_rl.thinking_template import ASSISTANT_PREFIX
from scripts import evaluate_benchmarks as benchmark
from scripts import evaluate_dtpo as dtpo
from scripts.prepare_benchmarks import save_images
from sft.data import prepare_image as prepare_training_image


TOOL = ('<think>Read the detail.</think><tool_call>{"name":"request_local_region",'
        '"arguments":{"bbox_2d":[0,0,1000,1000]}}</tool_call>')
ANSWER = "<think>The answer is visible.</think><answer>42</answer>"


class Grid:
    def __init__(self, tokens):
        self.tokens = tokens

    def prod(self):
        return self

    def item(self):
        return self.tokens * 4


class ImageProcessor:
    merge_size = 2

    def __call__(self, images, *, return_tensors):
        return {"image_grid_thw": [
            Grid(max(1, round(im.width / 32)) * max(1, round(im.height / 32)))
            for im in images
        ]}


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n" + ASSISTANT_PREFIX

    def encode(self, text, *, add_special_tokens):
        return [1] * (text.count("<|image_pad|>") + len(text.split()))


class RecordingLLM:
    def __init__(self, *, use_tool, final_response):
        self.use_tool = use_tool
        self.final_response = final_response
        self.calls = []

    def generate(self, requests, **kwargs):
        self.calls.append(requests)
        response = TOOL if self.use_tool and len(self.calls) == 1 else self.final_response
        return [SimpleNamespace(outputs=[SimpleNamespace(text=response)]) for _ in requests]


class CPUEvaluator(dtpo.VLLMEvaluator):
    """Keep VLLMEvaluator.generate intact; replace only the model backend."""

    def __init__(self, *, use_tool=False, final_response=ANSWER):
        self.processor = SimpleNamespace(
            image_processor=ImageProcessor(), image_token="<|image_pad|>", tokenizer=Tokenizer()
        )
        self.max_prompt_length = 8192
        self.sampling_params = object()
        self.lora_request = None
        self.llm = RecordingLLM(use_tool=use_tool, final_response=final_response)


class TrainInferConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.high = root / "high.png"
        self.low = root / "low.png"
        # These shapes change again if min-pixel normalization is repeated.
        Image.linear_gradient("L").resize((661, 71)).convert("RGB").save(self.high)
        Image.linear_gradient("L").resize((128, 45)).convert("RGB").save(self.low)

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self, workflow, evaluator, *, high_only=False):
        if workflow == "dtpo":
            sample = dtpo.EvalSample("sample", "What number?", ["42"], self.high,
                                     self.low, [], False, False)
            return dtpo.evaluate_batch(evaluator, [sample], coverage_weight=0.5)[0][0]
        sample = benchmark.BenchmarkSample("ChartQA:sample", "ChartQA", "What number?",
                                           ["42"], self.high, self.low, None)
        mode = "high_only" if high_only else "low_tool"
        return benchmark.evaluate_batch(evaluator, [sample], mode)[0][0]

    def assert_image_matches_training(self, actual, path):
        with Image.open(path) as raw:
            expected = prepare_training_image(raw)
        self.assertEqual(actual.size, expected.size)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_both_evaluators_match_training_images_on_both_turns(self):
        for workflow in ("dtpo", "benchmark"):
            with self.subTest(workflow=workflow):
                evaluator = CPUEvaluator(use_tool=True)
                record = self.evaluate(workflow, evaluator)
                self.assertTrue(record["correct"])
                first, second = [call[0]["multi_modal_data"]["image"] for call in evaluator.llm.calls]
                self.assert_image_matches_training(first[0], self.low)
                self.assert_image_matches_training(second[0], self.low)
                self.assert_image_matches_training(second[1], self.high)
                self.assertEqual(second[1].size, (781, 83))

    def test_high_only_also_normalizes_the_raw_image_once(self):
        evaluator = CPUEvaluator()
        self.evaluate("benchmark", evaluator, high_only=True)
        actual = evaluator.llm.calls[0][0]["multi_modal_data"]["image"][0]
        self.assert_image_matches_training(actual, self.high)

    def test_malformed_final_answers_receive_no_answer_credit(self):
        malformed = (
            "<answer>42</answer>",
            "<think></think><answer>42</answer>",
            ANSWER + " extra text",
        )
        for workflow in ("dtpo", "benchmark"):
            for use_tool in (False, True):
                for response in malformed:
                    with self.subTest(workflow=workflow, use_tool=use_tool, response=response):
                        evaluator = CPUEvaluator(use_tool=use_tool, final_response=response)
                        record = self.evaluate(workflow, evaluator)
                        self.assertFalse(record["final_answer_valid"])
                        self.assertFalse(record["correct"])
                        self.assertIsNone(record["prediction"])
                        self.assertEqual(record["answer_score"], 0)

    def test_exif_high_image_matches_prepared_low_image_coordinates(self):
        source = Image.linear_gradient("L").resize((64, 96)).convert("RGB")
        exif = Image.Exif()
        exif[274] = 6
        source.save(self.high, format="JPEG", exif=exif)
        raw_bytes = self.high.read_bytes()
        root = Path(self.temporary.name)
        high_rel, low_rel, high_size, low_size, _ = save_images(raw_bytes, root)
        loaded = dtpo.load_rgb(root / high_rel)
        with Image.open(root / high_rel) as stored:
            expected = ImageOps.exif_transpose(stored).convert("RGB")
        self.assertEqual(loaded.size, tuple(high_size))
        self.assertEqual(loaded.tobytes(), expected.tobytes())
        self.assertEqual(loaded.size, (96, 64))
        with Image.open(root / low_rel) as low:
            self.assertEqual(low.size, tuple(low_size))
        action = dtpo.parse_action(
            TOOL.replace("[0,0,1000,1000]", "[500,0,1000,1000]"),
            allow_tool=True, image_size=tuple(low_size),
        )
        crop, _ = dtpo.execute_crop(loaded, tuple(low_size), action.bbox)
        self.assertEqual(crop.tobytes(), expected.crop((48, 0, 96, 64)).tobytes())
        self.assertEqual((root / high_rel).read_bytes(), raw_bytes)


if __name__ == "__main__":
    unittest.main()

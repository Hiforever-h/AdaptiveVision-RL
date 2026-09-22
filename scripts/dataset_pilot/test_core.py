import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from .annotate import annotate_row, generation_config, validate_annotation
from .common import answer_check, digest, pixel_box, validate_box
from .reward import geometry_reward


class GeometryTests(unittest.TestCase):
    def test_exact_disjoint_and_oversized_crop(self):
        reference = [[0.1, 0.1, 0.3, 0.3]]
        self.assertAlmostEqual(geometry_reward(reference[0], reference)["reward"], 1)
        self.assertEqual(geometry_reward([0.6, 0.6, 0.9, 0.9], reference)["reward"], 0)
        oversized = geometry_reward([0, 0, 1, 1], reference)
        self.assertEqual(oversized["coverage"], 1)
        self.assertAlmostEqual(oversized["reward"], 0.2)

    def test_partial_evidence_and_missing_label(self):
        score = geometry_reward([0.2, 0.2, 0.4, 0.3], [[0.2, 0.2, 0.4, 0.4]])
        self.assertAlmostEqual(score["coverage"], 0.5)
        self.assertAlmostEqual(score["iou"], 0.5)
        self.assertAlmostEqual(score["reward"], 0.5)
        self.assertIsNone(geometry_reward([0, 0, 1, 1], []))

    def test_alternative_references_not_union(self):
        refs = [[0, 0, 0.2, 0.2], [0.8, 0.8, 1, 1]]
        self.assertEqual(geometry_reward(refs[1], refs)["reward"], 1)

    def test_bad_coordinates_and_pixel_rounding(self):
        for box in [[0, 0, 0, 1], [0, 0, float("nan"), 1], [-1, 0, 1, 1], [True, 0, 1, 1]]:
            with self.assertRaises(ValueError):
                validate_box(box)
        self.assertEqual(pixel_box([0.11, 0.11, 0.91, 0.91], 11, 11), [1, 1, 11, 11])


class AnnotationTests(unittest.TestCase):
    def test_answer_matching_preserves_units_and_no_substring_matching(self):
        self.assertTrue(answer_check("42,138", ["42138"])["match"])
        self.assertTrue(answer_check(" LEIGH  BARDUGO ", ["Leigh Bardugo"])["match"])
        for pred, gold in [("5", "15"), ("5%", "5"), ("5 kg", "5"), ("0.05", "5"), (".5", "5"), ("Friday or Sunday", "Friday")]:
            self.assertFalse(answer_check(pred, [gold])["match"])

    def test_annotation_schema(self):
        valid = {"status": "localized", "reference_boxes": [[0, 0, 0.5, 0.5]], "answer_from_image": "5"}
        validate_annotation(valid)
        self.assertEqual(validate_annotation({**valid, "type": "json_object"}), valid)
        with self.assertRaises(ValueError):
            validate_annotation({**valid, "evidence_description": "unwanted"})
        with self.assertRaises(ValueError):
            validate_annotation({**valid, "reference_boxes": []})

    def test_pixel_annotation_schema(self):
        value = {"status": "localized", "reference_boxes": [[10, 20, 90, 80]],
                 "answer_from_image": "5"}
        self.assertEqual(validate_annotation(value, 100, 100, "pixels"), value)
        with self.assertRaises(ValueError):
            validate_annotation({**value, "reference_boxes": [[10, 20, 101, 80]]},
                                100, 100, "pixels")
        with self.assertRaises(ValueError):
            validate_annotation({**value, "reference_boxes": [[10.5, 20, 90, 80]]},
                                100, 100, "pixels")

    def test_request_hides_reference_and_resume_avoids_duplicate_billing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "image.png").write_bytes(b"test-image")
            row = {"sample_id": "one", "question": "read the number", "answers": ["secret-gold"],
                   "image_path": "image.png", "image_sha256": digest(b"test-image"), "width": 100, "height": 100}
            response = Mock(status_code=200)
            response.json.return_value = {"model": "deepseek-flash", "choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps({"status": "localized", "reference_boxes": [[0, 0, 0.5, 0.5]], "answer_from_image": "7"})}}], "usage": {"total_tokens": 10}}
            with patch("scripts.dataset_pilot.annotate.requests.post", return_value=response) as post:
                result = annotate_row(row, output, "fake-api-key", "deepseek-flash", "prompt")
                sent = post.call_args.kwargs
                self.assertNotIn("secret-gold", json.dumps(sent["json"]))
                self.assertFalse(sent["allow_redirects"])
                self.assertTrue(result["quality_checks"]["valid_annotation"])
                annotate_row(row, output, "fake-api-key", "deepseek-flash", "prompt")
                self.assertEqual(post.call_count, 1)
                self.assertNotIn("fake-api-key", (output / "api_cache/one.json").read_text())

    def test_zai_request_uses_official_endpoint_and_supported_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "image.png").write_bytes(b"test-image")
            row = {"sample_id": "one", "question": "read the number", "answers": ["7"],
                   "image_path": "image.png", "image_sha256": digest(b"test-image"),
                   "width": 100, "height": 100}
            response = Mock(status_code=200)
            response.json.return_value = {"model": "glm-5.3-flash", "choices": [{
                "finish_reason": "stop", "message": {"content": json.dumps({
                    "status": "localized", "reference_boxes": [[10, 20, 90, 80]],
                    "answer_from_image": "7"})}}], "usage": {"total_tokens": 10}}
            with patch("scripts.dataset_pilot.annotate.requests.post", return_value=response) as post:
                result = annotate_row(row, output, "fake-zai-key", "glm-5.3-flash", "prompt",
                                      coordinate_format="pixels", provider="zai")
                endpoint = post.call_args.args[0]
                payload = post.call_args.kwargs["json"]
                self.assertEqual(endpoint, "https://api.z.ai/api/paas/v4/chat/completions")
                self.assertEqual(payload["thinking"]["type"], "enabled")
                self.assertEqual(payload["reasoning_effort"], "max")
                self.assertEqual(result["annotation_meta"]["provider"], "zai")
                self.assertEqual(result["region_annotation"]["reported_reference_boxes_pixels"],
                                 [[10, 20, 90, 80]])
                self.assertNotIn("fake-zai-key", (output / "api_cache/one.json").read_text())

    def test_zai_disabled_thinking_probe_has_no_reasoning_effort(self):
        config = generation_config("zai", thinking_mode="disabled")
        self.assertEqual(config["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", config)


if __name__ == "__main__":
    unittest.main()

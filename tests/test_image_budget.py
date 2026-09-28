import math
import unittest

from PIL import Image

from adaptive_vision_rl.verl.image_budget import fit_image_prompt, pad_thin_crop


class FakeGrid:
    def __init__(self, patch_count):
        self.patch_count = patch_count

    def prod(self):
        return self

    def item(self):
        return self.patch_count


class FakeImageProcessor:
    merge_size = 2

    def __call__(self, images, *, return_tensors):
        assert return_tensors == "pt"
        return {
            "image_grid_thw": [
                FakeGrid(4 * math.ceil(image.width / 28) * math.ceil(image.height / 28))
                for image in images
            ]
        }


class FakeProcessor:
    image_token = "<|image_pad|>"
    image_processor = FakeImageProcessor()


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        image_tokens = text.count("<|image_pad|>")
        other_text = text.replace("<|image_pad|>", "")
        return [1] * (image_tokens + len(other_text.split()))


def process_image(image, *, max_pixels):
    if image.width * image.height <= max_pixels:
        return image
    factor = math.sqrt(max_pixels / (image.width * image.height))
    return image.resize((max(1, int(image.width * factor)), max(1, int(image.height * factor))))


class ImageBudgetTests(unittest.TestCase):
    def test_extremely_thin_crop_is_padded_without_losing_pixels(self):
        image = Image.new("RGB", (1, 400), "white")
        padded = pad_thin_crop(image)
        self.assertEqual(padded.size, (4, 400))
        self.assertEqual(padded.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(padded.getpixel((1, 0)), (255, 255, 255))

    def test_keeps_images_unchanged_when_prompt_fits(self):
        images = [Image.new("RGB", (256, 256))]
        fitted = fit_image_prompt(
            prompt="Question <image>",
            images=images,
            tokenizer=FakeTokenizer(),
            processor=FakeProcessor(),
            max_prompt_length=8192,
            process_image=process_image,
        )
        self.assertEqual(fitted.images[0].size, (256, 256))
        self.assertEqual(fitted.prompt_length, fitted.initial_prompt_length)

    def test_downsizes_only_when_two_image_prompt_overflows(self):
        images = [Image.new("RGB", (2048, 2048)) for _ in range(2)]
        fitted = fit_image_prompt(
            prompt="Question <image> Crop <image>",
            images=images,
            tokenizer=FakeTokenizer(),
            processor=FakeProcessor(),
            max_prompt_length=8192,
            process_image=process_image,
        )
        self.assertGreater(fitted.initial_prompt_length, 8192)
        self.assertLessEqual(fitted.prompt_length, 8192)
        self.assertTrue(any(image.size != (2048, 2048) for image in fitted.images))
        self.assertEqual(len(fitted.vision_tokens), 2)
        self.assertEqual(fitted.expanded_prompt.count("<|image_pad|>"), sum(fitted.vision_tokens))

    def test_reports_when_even_minimum_images_cannot_fit(self):
        with self.assertRaisesRegex(ValueError, "cannot fit max_prompt_length"):
            fit_image_prompt(
                prompt="Question <image>",
                images=[Image.new("RGB", (256, 256))],
                tokenizer=FakeTokenizer(),
                processor=FakeProcessor(),
                max_prompt_length=10,
                process_image=process_image,
            )


if __name__ == "__main__":
    unittest.main()

"""Fit multimodal observations into the actor's prompt token budget."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, Sequence

from PIL import Image


_MAX_IMAGE_PIXELS = 2048 * 2048
_MIN_IMAGE_PIXELS = 256 * 256


def pad_thin_crop(image: Image.Image) -> Image.Image:
    """Keep a tool crop inside Qwen-VL's image aspect-ratio limit."""

    max_aspect = 100
    target_width = max(image.width, ceil(image.height / max_aspect))
    target_height = max(image.height, ceil(image.width / max_aspect))
    if (target_width, target_height) == image.size:
        return image
    padded = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    padded.paste(image, ((target_width - image.width) // 2, (target_height - image.height) // 2))
    return padded


@dataclass(frozen=True)
class FittedImagePrompt:
    images: list[Image.Image]
    image_inputs: dict[str, Any]
    image_grid_thw: Any
    expanded_prompt: str
    vision_tokens: list[int]
    prompt_length: int
    initial_prompt_length: int


def fit_image_prompt(
    *,
    prompt: str,
    images: Sequence[Any],
    tokenizer: Any,
    processor: Any,
    max_prompt_length: int,
    process_image: Callable[..., Image.Image],
) -> FittedImagePrompt:
    """Reduce image resolution only when the expanded prompt would overflow.

    The actor needs one token for every image patch, while vLLM receives the
    compact image placeholders. Both paths must use the same processed images.
    """

    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")
    if prompt.count("<image>") != len(images):
        raise ValueError(
            f"prompt/image mismatch: {prompt.count('<image>')} placeholders for "
            f"{len(images)} images"
        )
    if not images:
        raise ValueError("fit_image_prompt requires at least one image")

    max_pixels = [_MAX_IMAGE_PIXELS] * len(images)
    initial_length = 0
    for attempt in range(32):
        processed = [
            process_image(image, max_pixels=pixels)
            for image, pixels in zip(images, max_pixels, strict=True)
        ]
        image_inputs = dict(processor.image_processor(processed, return_tensors="pt"))
        grids = image_inputs["image_grid_thw"]
        if len(grids) != len(processed):
            raise ValueError(
                f"image processor returned {len(grids)} grids for {len(processed)} images"
            )
        merge_length = int(processor.image_processor.merge_size) ** 2
        vision_tokens = [int(grid.prod().item()) // merge_length for grid in grids]
        expanded = prompt
        for count in vision_tokens:
            expanded = expanded.replace(
                "<image>",
                "<|vision_start|>"
                + processor.image_token * count
                + "<|vision_end|>",
                1,
            )
        prompt_length = len(tokenizer.encode(expanded, add_special_tokens=False))
        if attempt == 0:
            initial_length = prompt_length
        if prompt_length <= max_prompt_length:
            return FittedImagePrompt(
                images=processed,
                image_inputs=image_inputs,
                image_grid_thw=grids,
                expanded_prompt=expanded,
                vision_tokens=vision_tokens,
                prompt_length=prompt_length,
                initial_prompt_length=initial_length,
            )

        candidates = [
            index for index, pixels in enumerate(max_pixels)
            if pixels > _MIN_IMAGE_PIXELS
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: vision_tokens[item])
        current_area = processed[index].width * processed[index].height
        excess = prompt_length - max_prompt_length
        target_fraction = min(
            0.75,
            max(0.4, 1.0 - (excess + 64) / max(vision_tokens[index], 1)),
        )
        max_pixels[index] = max(
            _MIN_IMAGE_PIXELS,
            min(max_pixels[index] - 1, int(current_area * target_fraction)),
        )

    raise ValueError(
        f"multimodal prompt cannot fit max_prompt_length={max_prompt_length}; "
        f"initial_length={initial_length}, last_length={prompt_length}, "
        f"image_count={len(images)}"
    )

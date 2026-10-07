"""Shared image loading and raw-image normalization for SFT, rollout and inference."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def prepare_image(
    image: Any,
    *,
    max_pixels: int = 2048 * 2048,
    min_pixels: int = 256 * 256,
) -> Image.Image:
    """Normalize a raw PIL image or uint8 HWC observation exactly once.

    Keep the original floor-based sizing used by the trained model. Integer
    rounding can leave small images below min_pixels, so callers must pass raw
    images rather than feed a normalized result through this function again.
    """

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    result = ImageOps.exif_transpose(image).convert("RGB")
    area = result.width * result.height
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
    elif area < min_pixels:
        scale = math.sqrt(min_pixels / area)
    else:
        return result
    return result.resize((int(result.width * scale), int(result.height * scale)))

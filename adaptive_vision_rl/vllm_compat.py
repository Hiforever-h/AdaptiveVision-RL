"""Narrow compatibility fixes for the pinned vLLM release."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any


logger = logging.getLogger(__name__)

_BROKEN_CONNECTOR = ["model.visual.merger"]
_BROKEN_TOWER = ["model.visual."]
_FIXED_CONNECTOR = ["visual.merger"]
_FIXED_TOWER = ["visual."]


def _patch_qwen3vl_mapping(model_cls: type, mapping_cls: type) -> str:
    """Backport vLLM 0.11.2's Qwen3-VL module prefixes.

    Returns a small status string so the version-gated entry point can log and
    tests can verify the patch without importing vLLM.
    """

    current = model_cls.get_mm_mapping(None)
    connector = list(current.connector)
    tower = list(current.tower_model)
    if connector == _FIXED_CONNECTOR and tower == _FIXED_TOWER:
        return "already_fixed"
    if connector != _BROKEN_CONNECTOR or tower != _BROKEN_TOWER:
        return "unexpected_mapping"

    def get_mm_mapping(_self: Any):
        return mapping_cls.from_string_field(
            language_model="language_model",
            connector="visual.merger",
            tower_model="visual.",
        )

    model_cls.get_mm_mapping = get_mm_mapping
    return "patched"


def install_qwen3vl_lora_mapping_backport() -> bool:
    """Install the Qwen3-VL LoRA mapping fix only for vLLM 0.11.0.

    vLLM 0.11.0 prefixes the vision tower with ``model.visual`` even though
    Qwen3-VL registers it as ``visual``. Its LoRA manager consequently wraps
    visual layers and can fail during multimodal profiling. vLLM 0.11.2 fixes
    the two prefixes; this function carries only that fix while the project
    remains pinned to 0.11.0.
    """

    try:
        installed_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return False
    if installed_version.partition("+")[0] != "0.11.0":
        return False

    from vllm.model_executor.models.module_mapping import MultiModelKeys
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    status = _patch_qwen3vl_mapping(
        Qwen3VLForConditionalGeneration,
        MultiModelKeys,
    )
    if status == "unexpected_mapping":
        mapping = Qwen3VLForConditionalGeneration.get_mm_mapping(None)
        raise RuntimeError(
            "Refusing to overwrite an unknown vLLM 0.11.0 Qwen3-VL mapping: "
            f"connector={mapping.connector}, tower_model={mapping.tower_model}"
        )
    if status == "patched":
        logger.warning(
            "Applied the vLLM 0.11.2 Qwen3-VL LoRA mapping backport "
            "(connector=visual.merger, tower_model=visual.)."
        )
    return status in {"patched", "already_fixed"}

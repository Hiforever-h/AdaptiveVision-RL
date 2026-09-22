"""Annotate the frozen pilot with DeepSeek; cache every response before parsing."""
import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from .common import (DEFAULT_OUTPUT, ROOT, answer_check, digest, pixel_box,
                     read_jsonl, validate_box, write_json, write_jsonl)

PROVIDERS = {
    "deepseek": {
        "endpoint": "https://api.deepseek.com/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-flash",
    },
    "zai": {
        "endpoint": "https://api.z.ai/api/paas/v4/chat/completions",
        "key_env": "ZAI_API_KEY",
        "model_env": "ZAI_MODEL",
        "default_model": "glm-5.3-flash",
    },
}
PROMPT_FILE = Path(__file__).with_name("prompt.txt")


def validate_annotation(value, width=None, height=None, coordinate_format="normalized"):
    # Flash sometimes echoes response_format.type inside otherwise valid JSON.
    # Accept only this known transport annotation; preserve it in the raw cache.
    if isinstance(value, dict) and value.get("type") == "json_object":
        value = {k: v for k, v in value.items() if k != "type"}
    if not isinstance(value, dict) or set(value) != {"status", "reference_boxes", "answer_from_image"}:
        raise ValueError("Unexpected JSON fields")
    if value["status"] not in {"localized", "global", "uncertain"}:
        raise ValueError("Invalid annotation status")
    if not isinstance(value["answer_from_image"], str):
        raise ValueError("Answer must be a string")
    boxes = value["reference_boxes"]
    if not isinstance(boxes, list):
        raise ValueError("reference_boxes must be a list")
    if value["status"] == "localized" and len(boxes) != 1:
        raise ValueError("The pilot expects exactly one local box")
    if value["status"] != "localized" and boxes:
        raise ValueError("Non-local annotations must have no boxes")
    if coordinate_format == "normalized":
        checked = [validate_box(box) for box in boxes]
    elif coordinate_format == "pixels":
        if not width or not height:
            raise ValueError("Image dimensions are required for pixel coordinates")
        checked = []
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("pixel box must contain four coordinates")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box):
                raise ValueError("pixel coordinates must be numbers")
            if any(not float(v).is_integer() for v in box):
                raise ValueError("pixel coordinates must be whole numbers")
            x1, y1, x2, y2 = [int(v) for v in box]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("pixel box is outside the original image or has no area")
            checked.append([x1, y1, x2, y2])
    else:
        raise ValueError(f"Unsupported coordinate format: {coordinate_format}")
    return {**value, "reference_boxes": checked}


def generation_config(provider, thinking_mode=None, reasoning_effort=None):
    if provider == "deepseek":
        mode = thinking_mode or "disabled"
        config = {"temperature": 0, "thinking": {"type": mode},
                "max_tokens": 1024, "response_format": {"type": "json_object"}}
        if reasoning_effort and mode == "enabled":
            config["reasoning_effort"] = reasoning_effort
        return config
    if provider == "zai":
        # The vendor documents enabled as the only supported mode for
        # GLM-5.3-Flash. An explicit override is retained so unsupported modes
        # can be probed in a separately cached experiment without ambiguity.
        mode = thinking_mode or "enabled"
        config = {"temperature": 0, "thinking": {"type": mode},
                  "max_tokens": 4096, "response_format": {"type": "json_object"}}
        if mode == "enabled":
            config["thinking"]["clear_thinking"] = True
            config["reasoning_effort"] = reasoning_effort or "max"
        return config
    raise ValueError(f"Unsupported provider: {provider}")


def annotate_row(row, output, key, model, prompt, retry_errors=False,
                 coordinate_format="normalized", provider="deepseek",
                 thinking_mode=None, reasoning_effort=None):
    provider_config = PROVIDERS[provider]
    endpoint = provider_config["endpoint"]
    generation = generation_config(provider, thinking_mode, reasoning_effort)
    image_bytes = (output / row["image_path"]).read_bytes()
    if digest(image_bytes) != row["image_sha256"]:
        raise ValueError("Image differs from frozen manifest")
    text = f"Original image size: {row['width']} x {row['height']} pixels.\nQuestion:\n{row['question']}"
    config = {"model": model, "endpoint": endpoint, "prompt_sha256": digest(prompt.encode()),
              "image_sha256": row["image_sha256"], "question_sha256": digest(text.encode()),
              **generation, "detail": "original",
              "coordinate_format": coordinate_format}
    if provider != "deepseek":
        config["provider"] = provider
    fingerprint = digest(json.dumps(config, sort_keys=True).encode())
    cache = output / "api_cache" / f"{row['sample_id']}.json"
    if cache.exists():
        record = json.loads(cache.read_text())
        if record["fingerprint"] != fingerprint:
            raise ValueError("Cached request differs; use a new output directory for changed prompts/models")
        if record.get("response") or not retry_errors:
            return parse_record(row, record)
    payload = {"model": model, **generation,
               "messages": [
                   {"role": "system", "content": prompt},
                   {"role": "user", "content": [
                       {"type": "text", "text": text},
                       {"type": "image_url", "image_url": {
                           "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode(),
                           "detail": "original"}}]}]}
    record = {"sample_id": row["sample_id"], "fingerprint": fingerprint, "request_config": config,
              "created_at": datetime.now(timezone.utc).isoformat(), "attempts": []}
    if cache.exists():
        # Keep prior failures when --retry-errors is explicitly requested.
        record["prior_record"] = json.loads(cache.read_text())
    started = time.perf_counter()
    for attempt in range(3):
        try:
            response = requests.post(endpoint, json=payload,
                                     headers={"Authorization": f"Bearer {key}"},
                                     timeout=(15, 120), allow_redirects=False)
        except requests.RequestException as exc:
            # A timeout may already be billed: do not automatically replay it.
            record["error"] = {"kind": type(exc).__name__, "message": "Request failed; details omitted to protect credentials"}
            record["attempts"].append({"attempt": attempt + 1, "error": type(exc).__name__})
            break
        record["attempts"].append({"attempt": attempt + 1, "http_status": response.status_code})
        if response.status_code == 200:
            try:
                record["response"] = response.json()
            except ValueError:
                record["error"] = {"kind": "InvalidResponseJSON"}
            break
        record["error"] = {"kind": "HTTPError", "http_status": response.status_code}
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
            break
        time.sleep(2 ** (attempt + 1))
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    if "response" in record:
        record.pop("error", None)
    # Never store request headers, authorization, environment or image base64.
    write_json(cache, record)
    return parse_record(row, record)


def parse_record(row, record):
    result = {**row, "annotation_meta": {
        "provider": record["request_config"].get("provider", "deepseek"),
        "requested_model": record["request_config"]["model"],
        "prompt_sha256": record["request_config"]["prompt_sha256"],
        "request_fingerprint": record["fingerprint"], "created_at": record["created_at"],
        "elapsed_seconds": record["elapsed_seconds"], "attempts": record["attempts"],
        "reference_answer_in_prompt": False, "crop_only_validation": False,
        "raw_response_path": f"api_cache/{row['sample_id']}.json"},
        "quality_checks": {"answer_match": None, "valid_annotation": False,
                           "accepted_by_answer": False, "region_reward_eligible": False,
                           "localization_review": "unreviewed"}}
    if "response" not in record:
        result["annotation_error"] = record.get("error", {"kind": "MissingResponse"})
        return result
    response = record["response"]
    result["annotation_meta"].update({"returned_model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "usage": response.get("usage", {}), "response_id": response.get("id")})
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Response did not finish normally")
        raw_value = json.loads(choice["message"]["content"])
        coordinate_format = record["request_config"].get("coordinate_format", "normalized")
        value = validate_annotation(raw_value, row["width"], row["height"], coordinate_format)
        result["quality_checks"]["schema_normalizations"] = (
            ["removed_redundant_type_json_object"] if raw_value.get("type") == "json_object" else [])
        if coordinate_format == "pixels":
            reported_pixels = value["reference_boxes"]
            normalized_boxes = [[b[0] / row["width"], b[1] / row["height"],
                                 b[2] / row["width"], b[3] / row["height"]]
                                for b in reported_pixels]
            result["region_annotation"] = {**value,
                "reference_boxes": normalized_boxes,
                "reported_reference_boxes_pixels": reported_pixels}
            executed_boxes = reported_pixels
            coordinate_system = "model reported integer xyxy pixels on original; right/bottom exclusive; normalized deterministically for reward"
        else:
            result["region_annotation"] = value
            executed_boxes = [pixel_box(b, row["width"], row["height"]) for b in value["reference_boxes"]]
            coordinate_system = "xyxy_normalized_original; pixel boxes floor/ceil, right/bottom exclusive"
        result["region_annotation"]["reference_boxes_pixels"] = executed_boxes
        result["region_annotation"]["coordinate_system"] = coordinate_system
        check = answer_check(value["answer_from_image"], row["answers"])
        result["quality_checks"].update({"valid_annotation": True, "answer_match": check,
            "accepted_by_answer": check["match"],
            "region_reward_eligible": check["match"] and value["status"] == "localized"})
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        result["annotation_error"] = {"kind": "AnnotationValidationError", "message": str(exc)}
    return result


def run(output, limit, workers, retry_errors, prompt_file=PROMPT_FILE,
        coordinate_format="normalized", provider="deepseek", model_override=None,
        thinking_mode=None, reasoning_effort=None, allow_large=False):
    load_dotenv(ROOT / ".env", override=False)
    provider_config = PROVIDERS[provider]
    key_env = provider_config["key_env"]
    key = os.environ.get(key_env, "").strip()
    if not key:
        raise SystemExit(f"{key_env} is missing from environment/project .env")
    model = model_override or os.environ.get(
        provider_config["model_env"], provider_config["default_model"])
    manifest = read_jsonl(output / "manifest.jsonl")
    if len(manifest) > 50 and not allow_large:
        raise SystemExit("Manifest has more than 50 rows; pass --allow-large for an authorized formal run")
    rows = manifest
    if limit is not None:
        rows = rows[:limit]
    prompt = prompt_file.read_text()
    snapshot = output / "prompt.txt"
    if snapshot.exists() and snapshot.read_text() != prompt:
        raise ValueError("Run prompt snapshot differs; choose a new output directory")
    if not snapshot.exists():
        snapshot.write_text(prompt)
    previous = output / "annotations.jsonl"
    results = {r["sample_id"]: r for r in read_jsonl(previous)} if previous.exists() else {}
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(annotate_row, row, output, key, model, prompt,
                               retry_errors, coordinate_format, provider,
                               thinking_mode, reasoning_effort): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results[result["sample_id"]] = result
            completed += 1
            check = result["quality_checks"]
            print(f"{completed}/{len(rows)} {result['sample_id']}: "
                  f"valid={check['valid_annotation']} answer_match={check['accepted_by_answer']}", flush=True)
            write_jsonl(output / "annotations.jsonl", [results[r["sample_id"]] for r in manifest if r["sample_id"] in results])
    if any("annotation_error" in r for r in results.values()):
        raise SystemExit("Some annotations failed; inspect cached errors before --retry-errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, choices=range(1, 51), default=3,
                        help="Concurrent requests; Z.AI documents a maximum of 50")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--allow-large", action="store_true",
                        help="Allow an authorized run whose manifest contains more than 50 rows")
    parser.add_argument("--prompt", type=Path, default=PROMPT_FILE)
    parser.add_argument("--coordinate-format", choices=["normalized", "pixels"], default="normalized")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="deepseek")
    parser.add_argument("--model", help="Override the provider's default model code")
    parser.add_argument("--thinking", choices=["enabled", "disabled"],
                        help="Override the provider's default thinking mode")
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"],
                        help="Reasoning effort when thinking is enabled")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.thinking == "disabled" and args.reasoning_effort:
        parser.error("--reasoning-effort cannot be used with --thinking disabled")
    run(args.output.resolve(), args.limit, args.workers, args.retry_errors,
        args.prompt.resolve(), args.coordinate_format, args.provider, args.model,
        args.thinking, args.reasoning_effort, args.allow_large)

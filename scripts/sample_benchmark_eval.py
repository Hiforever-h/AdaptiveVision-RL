"""Freeze a self-contained, balanced 3,000-question evaluation subset.

Usage from the repository root:
    python scripts/sample_benchmark_eval.py create
    python scripts/sample_benchmark_eval.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "eval_6bench_3000_v1"
STAGING = DATA / ".eval_6bench_3000_v1.staging"
SEED = 20260929
DATASETS = ("ChartQA", "OCRBench", "MME", "RealWorldQA", "POPE", "MathVerse")
TARGETS = {name: 500 for name in DATASETS}
TRAIN_MANIFESTS = (
    DATA / "visionthink_3000_300_500" / "manifest.jsonl",
    DATA / "visionthink_3000_300_500_balanced" / "manifest.jsonl",
    DATA / "pilot20_v6_glm53_pixels" / "manifest.jsonl",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def rng_for(name: str) -> random.Random:
    value = int.from_bytes(hashlib.sha256(f"{SEED}:{name}".encode()).digest()[:8], "big")
    return random.Random(value)


def select_unique(rows: list[dict], count: int, rng: random.Random, forbidden: set[str], used: set[str] | None = None) -> list[dict]:
    candidates = list(rows)
    rng.shuffle(candidates)
    used = used if used is not None else set()
    chosen = []
    for row in candidates:
        image_hash = row["image_sha256"]
        if image_hash in forbidden or image_hash in used:
            continue
        chosen.append(row)
        used.add(image_hash)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"only found {len(chosen)} of {count} unique images")
    return chosen


def balanced_quotas(categories: list[str], total: int, rng: random.Random) -> dict[str, int]:
    shuffled = sorted(categories)
    rng.shuffle(shuffled)
    quotas = {category: total // len(categories) for category in categories}
    for category in shuffled[: total % len(categories)]:
        quotas[category] += 1
    return quotas


def select_stratified(rows: list[dict], key: str, quota: dict[str, int], rng: random.Random, forbidden: set[str]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["data"][key])].append(row)
    used: set[str] = set()
    chosen = []
    # Scarce categories go first to avoid using an image shared by categories.
    for category in sorted(quota, key=lambda cat: (len({r['image_sha256'] for r in groups[cat]}), cat)):
        chosen.extend(select_unique(groups[category], quota[category], rng, forbidden, used))
    return chosen


def select_paired(rows: list[dict], quota_pairs: dict[str, int], rng: random.Random, forbidden: set[str]) -> list[dict]:
    groups: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        category = str(row["data"]["category"])
        answer = str(row["answer"]).strip().lower()
        groups[category][row["image_sha256"]][answer].append(row)
    used: set[str] = set()
    chosen = []
    for category in sorted(quota_pairs):
        image_hashes = sorted(groups[category])
        rng.shuffle(image_hashes)
        selected_groups = 0
        for image_hash in image_hashes:
            if image_hash in forbidden or image_hash in used:
                continue
            by_answer = groups[category][image_hash]
            if not by_answer["yes"] or not by_answer["no"]:
                continue
            chosen.extend((rng.choice(by_answer["yes"]), rng.choice(by_answer["no"])))
            used.add(image_hash)
            selected_groups += 1
            if selected_groups == quota_pairs[category]:
                break
        if selected_groups != quota_pairs[category]:
            raise RuntimeError(f"{category}: only {selected_groups}/{quota_pairs[category]} Yes/No image pairs")
    return chosen


def select_mathverse(rows: list[dict], rng: random.Random, forbidden: set[str]) -> list[dict]:
    problems: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        problem = str(row["data"]["problem_index"])
        version = str(row["data"]["problem_version"])
        if version in problems[problem]:
            raise RuntimeError(f"duplicate MathVerse problem/version: {problem}/{version}")
        problems[problem][version] = row
    versions = sorted({str(row["data"]["problem_version"]) for row in rows})
    if len(versions) != 5:
        raise RuntimeError(f"expected five MathVerse visual versions, found {versions}")
    problem_ids = sorted(problems)
    rng.shuffle(problem_ids)
    used_problems: set[str] = set()
    used_images: set[str] = set()
    chosen = []
    for version in versions:
        selected = 0
        for problem in problem_ids:
            row = problems[problem].get(version)
            if row is None or problem in used_problems:
                continue
            image_hash = row["image_sha256"]
            if image_hash in forbidden or image_hash in used_images:
                continue
            chosen.append(row)
            used_problems.add(problem)
            used_images.add(image_hash)
            selected += 1
            if selected == 100:
                break
        if selected != 100:
            raise RuntimeError(f"MathVerse {version}: only {selected}/100 distinct problems/images")
    return chosen


def select_all(source_rows: dict[str, list[dict]], forbidden: set[str]) -> dict[str, list[dict]]:
    selected = {}
    selected["ChartQA"] = select_stratified(
        source_rows["ChartQA"], "human_or_machine", {"0": 250, "1": 250}, rng_for("ChartQA"), forbidden
    )
    selected["OCRBench"] = select_stratified(
        source_rows["OCRBench"], "question_type",
        {str(row["data"]["question_type"]): 50 for row in source_rows["OCRBench"]},
        rng_for("OCRBench"), forbidden,
    )
    mme_categories = sorted({str(row["data"]["category"]) for row in source_rows["MME"]})
    mme_rng = rng_for("MME")
    selected["MME"] = select_paired(
        source_rows["MME"], balanced_quotas(mme_categories, 250, mme_rng), mme_rng, forbidden
    )
    selected["RealWorldQA"] = select_unique(source_rows["RealWorldQA"], 500, rng_for("RealWorldQA"), forbidden)
    pope_rng = rng_for("POPE")
    selected["POPE"] = select_paired(
        source_rows["POPE"], balanced_quotas(["adversarial", "popular", "random"], 250, pope_rng), pope_rng, forbidden
    )
    selected["MathVerse"] = select_mathverse(source_rows["MathVerse"], rng_for("MathVerse"), forbidden)
    for name, rows in selected.items():
        if len(rows) != TARGETS[name]:
            raise RuntimeError(f"{name}: selected {len(rows)} instead of {TARGETS[name]}")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise RuntimeError(f"{name}: duplicate source sample IDs")
    return selected


def training_hashes() -> tuple[set[str], set[str]]:
    byte_hashes, pixel_hashes = set(), set()
    for path in TRAIN_MANIFESTS:
        for row in load_jsonl(path):
            byte_hashes.update(hash_value for key in ("source_bytes_sha256", "image_sha256") if (hash_value := row.get(key)))
            if row.get("pixel_sha256"):
                pixel_hashes.add(row["pixel_sha256"])
    return byte_hashes, pixel_hashes


def pixel_hash(path: Path) -> str:
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        prefix = f"{image.width}x{image.height}:".encode()
        return hashlib.sha256(prefix + image.tobytes()).hexdigest()


def pixel_collisions(selected: dict[str, list[dict]], training_pixels: set[str]) -> set[str]:
    collisions = set()
    cache: dict[str, str] = {}
    for name, rows in selected.items():
        for row in rows:
            image_hash = row["image_sha256"]
            if image_hash not in cache:
                cache[image_hash] = pixel_hash(DATA / name / row["high_image"])
            if cache[image_hash] in training_pixels:
                collisions.add(image_hash)
        print(f"{name}: checked {len(rows)} selected rows for pixel overlap", flush=True)
    return collisions


def answer_list(answer) -> list[str]:
    values = answer if isinstance(answer, list) else [answer]
    normalized = [str(value) for value in values if value is not None and str(value) != ""]
    if not normalized:
        raise RuntimeError("empty answer in selected row")
    return normalized


def materialize(selected: dict[str, list[dict]], source_hashes: dict[str, str], excluded_pixel_hashes: set[str]) -> None:
    if OUTPUT.exists() or STAGING.exists():
        raise FileExistsError(f"frozen output or staging already exists: {OUTPUT} / {STAGING}")
    STAGING.mkdir(parents=True)
    summary = {
        "name": OUTPUT.name,
        "seed": SEED,
        "total_questions": sum(map(len, selected.values())),
        "sampling_policy": "500 per dataset; balanced strata, paired Yes/No for MME and POPE, distinct base problems across MathVerse versions",
        "training_manifests_checked": [str(path.relative_to(ROOT)) for path in TRAIN_MANIFESTS],
        "exact_training_image_collisions": 0,
        "pixel_training_image_collisions": 0,
        "excluded_training_pixel_image_sha256": sorted(excluded_pixel_hashes),
        "datasets": {},
    }
    all_hashes: dict[str, set[str]] = {}
    for name in DATASETS:
        rows = sorted(selected[name], key=lambda row: row["sample_id"])
        source_dir, target_dir = DATA / name, STAGING / name
        target_dir.mkdir()
        source_metadata = json.loads((source_dir / "source.json").read_text())
        (target_dir / "upstream_source.json").write_text(json.dumps(source_metadata, ensure_ascii=False, indent=2) + "\n")
        with (target_dir / "samples.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                record = {
                    "eval_id": f"{name}:{row['sample_id']}",
                    "dataset": name,
                    "source_sample_id": row["sample_id"],
                    "split": row["split"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "answers": answer_list(row["answer"]),
                    "high_image": row["high_image"],
                    "low_image": row["low_image"],
                    "high_size": row["high_size"],
                    "low_size": row["low_size"],
                    "image_sha256": row["image_sha256"],
                    "source_file": row["source_file"],
                    "source_image_path": row["source_image_path"],
                    "source_data": row["data"],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                for field in ("high_image", "low_image"):
                    relative = Path(row[field])
                    destination = target_dir / relative
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_dir / relative, destination)
        strata_key = {
            "ChartQA": "human_or_machine", "OCRBench": "question_type", "MME": "category",
            "RealWorldQA": None, "POPE": "category", "MathVerse": "problem_version",
        }[name]
        strata = Counter(str(row["data"][strata_key]) if strata_key else "all" for row in rows)
        all_hashes[name] = {row["image_sha256"] for row in rows}
        summary["datasets"][name] = {
            "questions": len(rows),
            "unique_images": len(all_hashes[name]),
            "strata": dict(sorted(strata.items())),
            "source_manifest_sha256": source_hashes[name],
            "subset_manifest_sha256": digest(target_dir / "samples.jsonl"),
            "manifest": f"{name}/samples.jsonl",
        }
        print(f"{name}: copied {len(rows)} questions, {len(all_hashes[name])} image pairs", flush=True)
    cross = []
    for index, left in enumerate(DATASETS):
        for right in DATASETS[index + 1:]:
            shared = all_hashes[left] & all_hashes[right]
            if shared:
                cross.append({"datasets": [left, right], "shared_exact_images": len(shared)})
    summary["cross_dataset_exact_image_overlap"] = cross
    (STAGING / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (STAGING / "README.md").write_text(
        "# Frozen six-benchmark evaluation subset\n\n"
        "Each dataset has its own directory and `samples.jsonl`; image paths are relative to that directory. "
        "`images/high` contains unchanged source image bytes and `images/low` contains the paired half-width, half-height images. "
        "`selection.json` records the fixed seed, counts, strata, source manifest hashes, and overlap audit. "
        "The six datasets are combined only when computing a future overall metric.\n"
    )
    verify(STAGING)
    STAGING.replace(OUTPUT)
    print(f"Frozen evaluation subset: {OUTPUT}", flush=True)


def verify(directory: Path = OUTPUT) -> None:
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = None
    summary = json.loads((directory / "selection.json").read_text())
    total = 0
    for name in DATASETS:
        folder = directory / name
        manifest = folder / "samples.jsonl"
        rows = load_jsonl(manifest)
        if len(rows) != TARGETS[name] or len(rows) != summary["datasets"][name]["questions"]:
            raise RuntimeError(f"{name}: wrong question count")
        if digest(manifest) != summary["datasets"][name]["subset_manifest_sha256"]:
            raise RuntimeError(f"{name}: manifest checksum changed")
        seen_ids, seen_images = set(), set()
        for row in rows:
            if row["eval_id"] in seen_ids or not row["question"] or not row["answers"]:
                raise RuntimeError(f"{name}: bad sample ID, question, or answer")
            seen_ids.add(row["eval_id"])
            high_path, low_path = folder / row["high_image"], folder / row["low_image"]
            if not high_path.is_file() or not low_path.is_file():
                raise RuntimeError(f"{name}: missing image pair for {row['eval_id']}")
            if row["image_sha256"] not in seen_images:
                if digest(high_path) != row["image_sha256"]:
                    raise RuntimeError(f"{name}: high image changed for {row['eval_id']}")
                with Image.open(high_path) as high, Image.open(low_path) as low:
                    high_size = ImageOps.exif_transpose(high).size
                    low_size = tuple(max(1, side // 2) for side in high_size)
                    if high_size != tuple(row["high_size"]) or low.size != low_size or low.size != tuple(row["low_size"]):
                        raise RuntimeError(f"{name}: wrong image dimensions for {row['eval_id']}")
                seen_images.add(row["image_sha256"])
        total += len(rows)
        print(f"{name}: verified {len(rows)} questions, {len(seen_images)} image pairs", flush=True)
    if total != 3000 or summary["total_questions"] != total:
        raise RuntimeError(f"wrong overall count: {total}")
    print(f"Verified total: {total} questions", flush=True)


def create() -> None:
    if OUTPUT.exists() or STAGING.exists():
        raise FileExistsError(f"frozen output or staging already exists: {OUTPUT} / {STAGING}")
    source_rows = {name: load_jsonl(DATA / name / "samples.jsonl") for name in DATASETS}
    source_hashes = {name: digest(DATA / name / "samples.jsonl") for name in DATASETS}
    forbidden, training_pixels = training_hashes()
    print(f"Loaded {len(forbidden)} training image byte hashes and {len(training_pixels)} pixel hashes", flush=True)
    excluded_pixel_hashes: set[str] = set()
    for attempt in range(5):
        selected = select_all(source_rows, forbidden)
        collisions = pixel_collisions(selected, training_pixels)
        if not collisions:
            materialize(selected, source_hashes, excluded_pixel_hashes)
            return
        print(f"Pixel overlap on attempt {attempt + 1}: excluding {len(collisions)} images", flush=True)
        excluded_pixel_hashes.update(collisions)
        forbidden.update(collisions)
    raise RuntimeError("unable to form a subset without training-image overlap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    arguments = parser.parse_args()
    if arguments.action == "create":
        create()
    else:
        verify()

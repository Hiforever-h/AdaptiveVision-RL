"""Create a deterministic random annotation review gallery from a formal split."""
import argparse
import html
import json
import random
import textwrap
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .common import ROOT, read_jsonl, write_json, write_jsonl


def font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def clipped_text(value, width=72, lines=2):
    wrapped = textwrap.wrap(str(value).replace("\n", " "), width=width)
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1][:-1] + "…"
    return "\n".join(wrapped)


def build(source, report_dir, count, seed):
    manifest = read_jsonl(source / "manifest.jsonl")
    annotations = {row["sample_id"]: row for row in read_jsonl(source / "annotations.jsonl")}
    if count > len(manifest):
        raise ValueError(f"Requested {count} rows from a {len(manifest)}-row manifest")
    selected = sorted(random.Random(seed).sample(manifest, count),
                      key=lambda row: row["sample_id"])
    records = [annotations.get(row["sample_id"], row) for row in selected]
    assets = report_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cards, tiles = [], []
    stats = Counter()
    for row in records:
        sid = row["sample_id"]
        annotation = row.get("region_annotation")
        error = row.get("annotation_error")
        quality = row.get("quality_checks", {})
        original = Image.open(source / row["image_path"]).convert("RGB")
        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        boxes = annotation.get("reference_boxes_pixels", []) if annotation else []
        for box in boxes:
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1),
                           outline="#ef233c", width=max(3, round(original.width / 180)))
        overlay_path = assets / f"{sid}-overlay.png"
        overlay.save(overlay_path)
        crop_path = None
        if boxes:
            crop_path = assets / f"{sid}-crop.png"
            original.crop(boxes[0]).save(crop_path)
        status = annotation.get("status") if annotation else "error"
        stats[status] += 1
        stats["valid"] += int(annotation is not None)
        stats["answer_match"] += int(bool(quality.get("accepted_by_answer")))
        stats["use_tool_true"] += int(bool(row["source_info"].get("use_tool")))
        teacher = annotation.get("answer_from_image", "") if annotation else ""
        tile = Image.new("RGB", (1000, 760), "white")
        td = ImageDraw.Draw(tile)
        td.text((20, 15), f"{sid} | status={status} | hint={row['source_info']['use_tool']}",
                fill="#111827", font=font(24))
        td.multiline_text((20, 50), clipped_text("Q: " + row["question"]),
                          fill="#263247", font=font(20), spacing=5)
        td.text((20, 112), clipped_text("Gold: " + " / ".join(row["answers"]), 90, 1),
                fill="#0f5132", font=font(18))
        td.text((20, 138), clipped_text("GLM: " + teacher, 90, 1),
                fill="#7c2d12", font=font(18))
        view = ImageOps.contain(overlay, (620, 550), Image.Resampling.LANCZOS)
        tile.paste(view, (20 + (620 - view.width) // 2, 185 + (550 - view.height) // 2))
        if boxes:
            crop = original.crop(boxes[0])
            crop_view = ImageOps.contain(crop, (320, 500), Image.Resampling.LANCZOS)
            tile.paste(crop_view, (660 + (320 - crop_view.width) // 2,
                                   210 + (500 - crop_view.height) // 2))
            td.text((660, 185), "Crop", fill="#111827", font=font(18))
        elif error:
            td.multiline_text((660, 210), clipped_text(json.dumps(error, ensure_ascii=False), 30, 8),
                              fill="#991b1b", font=font(17), spacing=5)
        tiles.append(tile)
        crop_html = (f'<img src="assets/{crop_path.name}" alt="{sid} crop">'
                     if crop_path else "<p>No localized crop</p>")
        detail = html.escape(json.dumps(error or annotation or {}, ensure_ascii=False, indent=2))
        cards.append(f"""<article><h2>{html.escape(sid)}</h2>
<p><b>Question:</b> {html.escape(row['question'])}</p>
<p><b>Gold:</b> {html.escape(' / '.join(row['answers']))}<br>
<b>GLM:</b> {html.escape(teacher)}<br><b>Status:</b> {html.escape(status)} ·
<b>source use_tool:</b> {row['source_info'].get('use_tool')}</p>
<div class="images"><img src="assets/{overlay_path.name}" alt="{sid} overlay">{crop_html}</div>
<details><summary>Annotation JSON</summary><pre>{detail}</pre></details></article>""")
    for start in range(0, len(tiles), 4):
        sheet = Image.new("RGB", (2000, 1520), "#dfe4ec")
        for index, tile in enumerate(tiles[start:start + 4]):
            sheet.paste(tile, ((index % 2) * 1000, (index // 2) * 760))
        sheet.save(report_dir / f"contact-{start // 4 + 1:02d}.jpg", quality=94)
    summary = {
        "source": str(source), "source_count": len(manifest), "sample_count": count,
        "sample_seed": seed, **dict(stats),
        "selection": "uniform random sample without replacement from the full manifest",
    }
    write_json(report_dir / "summary.json", summary)
    write_jsonl(report_dir / "sample20.jsonl", records)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>
body{max-width:1280px;margin:30px auto;font:15px/1.55 system-ui;color:#162033;background:#f4f6fa}
article{background:white;padding:22px;margin:20px 0;border:1px solid #dce2ec;border-radius:12px}
.images{display:grid;grid-template-columns:3fr 2fr;gap:18px;align-items:start}.images img{max-width:100%;max-height:720px;object-fit:contain}
pre{white-space:pre-wrap}@media(max-width:700px){.images{grid-template-columns:1fr}}</style>
<h1>Val 300 · GLM-5.3-Flash max · 随机 20 条定位检查</h1>""" + "".join(cards) + "</html>"
    (report_dir / "index.html").write_text(page)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT / "reports/val300_glm53_max_sample20")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    build(args.source.resolve(), args.report_dir.resolve(), args.count, args.seed)

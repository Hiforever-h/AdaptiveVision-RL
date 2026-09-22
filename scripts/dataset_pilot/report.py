"""Build a local HTML review gallery, previews, and a factual pilot summary."""
import argparse
import base64
import html
import io
import json
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .common import DEFAULT_OUTPUT, ROOT, pixel_box, read_jsonl, write_json, write_jsonl
from .reward import area, geometry_reward


def embedded(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_report(output, report_dir):
    manifest = read_jsonl(output / "manifest.jsonl")
    run_name = output.name
    annotation_path = output / "annotations.jsonl"
    annotations = {r["sample_id"]: r for r in read_jsonl(annotation_path)} if annotation_path.exists() else {}
    records = [annotations.get(r["sample_id"], r) for r in manifest]
    review_path = report_dir / "visual_review.jsonl"
    reviews = {r["sample_id"]: r for r in read_jsonl(review_path)} if review_path.exists() else {}
    for row in records:
        if row["sample_id"] in reviews and row.get("region_annotation"):
            review = reviews[row["sample_id"]]
            if review["request_fingerprint"] != row["annotation_meta"]["request_fingerprint"]:
                raise ValueError("Visual review refers to a different annotation request")
            row["visual_review"] = review
            row["quality_checks"] = {**row["quality_checks"],
                "accepted_by_answer": review["answer_correct"],
                "acceptance_basis": "Codex visual review; original automatic result remains in answer_match",
                "region_reward_eligible": review["answer_correct"] and row["region_annotation"]["status"] == "localized",
                "localization_review": review["localization_verdict"]}
    previews = output / "previews"
    previews.mkdir(exist_ok=True)
    cards, tiles, areas, latencies, geometry = [], [], [], [], []
    stats = Counter()
    models, providers = set(), set()
    tokens = Counter()
    table = []
    for row in records:
        sid = row["sample_id"]
        annotation = row.get("region_annotation")
        quality = row.get("quality_checks", {})
        meta = row.get("annotation_meta", {})
        if meta.get("returned_model"):
            models.add(meta["returned_model"])
        if meta.get("provider"):
            providers.add(meta["provider"])
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"]:
            tokens[key] += meta.get("usage", {}).get(key, 0)
        if "elapsed_seconds" in meta:
            latencies.append(meta["elapsed_seconds"])
        if annotation:
            stats["valid_annotations"] += 1
            status = annotation["status"]
            stats[status] += 1
            stats["answers_matched"] += int(quality["answer_match"]["match"])
            stats["region_reward_eligible"] += int(quality["region_reward_eligible"])
            review = row.get("visual_review")
            if review:
                stats["visually_reviewed"] += 1
                stats["answers_correct_after_review"] += int(review["answer_correct"])
                stats["localization_" + review["localization_verdict"]] += 1
            display_status = "答案匹配" if quality["accepted_by_answer"] else "答案待核对"
            css_status = "pass" if quality["accepted_by_answer"] else "review"
            predicted = annotation["answer_from_image"]
            if review:
                display_status = "答案正确（Codex 复核）" if review["answer_correct"] else "答案不符（Codex 复核）"
                css_status = "pass" if review["answer_correct"] else "review"
        elif "annotation_error" in row:
            stats["errors"] += 1
            display_status = "API／标注错误"
            css_status = "error"
            predicted = "未获得有效标注"
        else:
            stats["pending"] += 1
            display_status = "待标注"
            css_status = "pending"
            predicted = "尚未调用 API"
        original = Image.open(output / row["image_path"]).convert("RGB")
        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        crop_html = "<p class='muted'>暂无局部框</p>"
        boxes = annotation.get("reference_boxes", []) if annotation else []
        for index, box in enumerate(boxes):
            px = pixel_box(box, row["width"], row["height"])
            draw.rectangle((px[0], px[1], px[2] - 1, px[3] - 1), outline="#ef4444", width=max(2, round(row["width"] / 250)))
            crop = original.crop(px)
            crop.save(previews / f"{sid}-crop-{index}.png")
            crop_html = f"<a href='{embedded(crop)}' target='_blank'><img src='{embedded(crop)}' alt='参考区域裁剪'></a>"
            areas.append(area(box))
            executed = [px[0] / row["width"], px[1] / row["height"], px[2] / row["width"], px[3] / row["height"]]
            geometry.append({"sample_id": sid, "reference_area_fraction": area(box),
                             "executed_reference_crop": geometry_reward(executed, boxes),
                             "whole_image_crop": geometry_reward([0, 0, 1, 1], boxes)})
        overlay.save(previews / f"{sid}-overlay.png")
        thumb = ImageOps.contain(overlay, (620, 450), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (660, 520), "white")
        tile.paste(thumb, ((660 - thumb.width) // 2, 55 + (450 - thumb.height) // 2))
        td = ImageDraw.Draw(tile)
        td.text((16, 12), f"{sid}  tool_hint={row['source_info']['use_tool']}", fill="black")
        td.text((16, 30), f"{row['width']}x{row['height']}   box={'yes' if boxes else 'none'}", fill="black")
        tiles.append(tile)
        box_text = json.dumps(boxes)
        extra = json.dumps(row.get("annotation_error", {}), ensure_ascii=False) if "annotation_error" in row else box_text
        review_html = ""
        if row.get("visual_review"):
            review = row["visual_review"]
            review_html = f"<p class='note'><b>定位目视检查：</b>{html.escape(review['localization_verdict'])}<br>{html.escape(review['note'])}</p>"
        area_text = f"{sum(area(b) for b in boxes):.1%}" if boxes else "—"
        cards.append(f"""<article class='card' data-status='{css_status}'>
<div class='card-head'><strong>{html.escape(sid)}</strong><span class='badge {css_status}'>{display_status}</span></div>
<p class='question'>{html.escape(row['question'])}</p>
<div class='answers'><div><small>标准答案</small><strong>{html.escape(' / '.join(row['answers']))}</strong></div>
<div><small>answer_from_image</small><strong>{html.escape(predicted)}</strong></div></div>
<p class='muted'>{row['width']} × {row['height']} · 原 use_tool={row['source_info']['use_tool']} · 框面积 {area_text}</p>
<div class='visuals'><figure><figcaption>原图与参考框</figcaption><a href='{embedded(overlay)}' target='_blank'><img src='{embedded(overlay)}' alt='{sid} 原图与参考框'></a></figure>
<figure><figcaption>裁剪预览（不要求独立可回答）</figcaption>{crop_html}</figure></div>
{review_html}<details><summary>坐标／状态详情</summary><pre>{html.escape(extra)}</pre></details></article>""")
        table.append(f"| {sid} | {display_status} | {html.escape(' / '.join(row['answers'])).replace('|', '/')} | {html.escape(predicted).replace('|', '/').replace(chr(10), ' ')} | {area_text} |")
    report_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(tiles), 4):
        sheet = Image.new("RGB", (1320, 1040), "#e5e7eb")
        for j, tile in enumerate(tiles[start:start + 4]):
            sheet.paste(tile, ((j % 2) * 660, (j // 2) * 520))
        sheet.save(previews / f"contact-{start // 4 + 1:02d}.jpg", quality=92)
    summary = {"run_name": run_name, "sample_count": len(records), **dict(stats),
               "providers": sorted(providers), "returned_models": sorted(models),
               "prompt_sha256": sorted({r["annotation_meta"]["prompt_sha256"] for r in records if "annotation_meta" in r}),
               "usage_reported_by_api": dict(tokens),
               "mean_latest_cached_request_seconds": statistics.mean(latencies) if latencies else None,
               "median_box_area_fraction": statistics.median(areas) if areas else None,
               "selection_criterion": "answer correctness (exact match or recorded Codex visual review); no crop-only verification",
               "localization_quality": "Codex image inspection, not human ground truth; separate from answer-only acceptance",
               "split_status": "pilot only; train 3000/dev 300 from Smart-Train, test 500 from Smart-Val not yet constructed"}
    write_json(report_dir / "summary.json", summary)
    write_jsonl(output / "geometry_diagnostics.jsonl", geometry)
    write_jsonl(output / "reviewed_annotations.jsonl", records)
    write_jsonl(output / "accepted_by_answer.jsonl", [r for r in records if
        r.get("visual_review", {}).get("answer_correct", r.get("quality_checks", {}).get("accepted_by_answer", False))])
    metrics = f"<div class='metric'><b>{len(records)}</b>样本</div><div class='metric'><b>{stats['valid_annotations']}</b>有效标注</div><div class='metric'><b>{stats['answers_matched']}</b>严格答案匹配</div><div class='metric'><b>{stats['answers_correct_after_review']}</b>复核后答案正确</div><div class='metric'><b>{stats['errors'] + stats['pending']}</b>错误／待运行</div>"
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>AdaptiveVision-RL · 20 条标注试验</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6fa;color:#192336;font:15px/1.6 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:32px 24px}h1{font-size:30px;margin:8px 0}.eyebrow{color:#526178;font-size:13px;letter-spacing:.12em}.note{padding:16px 20px;border-left:4px solid #4f46e5;background:#eef0ff;border-radius:8px}.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.metric{background:white;flex:1;min-width:145px;padding:16px 20px;border:1px solid #e2e6ef;border-radius:12px;color:#526178}.metric b{display:block;font-size:28px;color:#192336}.toolbar{display:flex;gap:12px;position:sticky;top:0;background:#f5f6faf5;padding:14px 0;z-index:1}input,select{font:inherit;padding:9px 12px;border:1px solid #cad2df;border-radius:8px}input{flex:1;min-width:0}.card{background:white;border:1px solid #e1e5ee;border-radius:14px;padding:22px;margin:18px 0}.card-head{display:flex;justify-content:space-between;gap:15px}.badge{border-radius:20px;padding:3px 12px;font-size:13px}.pass{background:#d9f5e5;color:#145c37}.review{background:#fff0c4;color:#7a500b}.error{background:#ffe1e1;color:#962727}.pending{background:#e8ecf3;color:#566175}.question{font-size:18px;white-space:pre-wrap}.answers{display:grid;grid-template-columns:1fr 1fr;gap:18px;background:#f5f7fb;padding:14px;border-radius:10px}.answers small{display:block;color:#617087}.answers strong{overflow-wrap:anywhere}.muted,figcaption{color:#637189;font-size:13px}.visuals{display:grid;grid-template-columns:1.5fr 1fr;gap:20px;align-items:start}figure{margin:0}figcaption{margin-bottom:8px}img{max-width:100%;max-height:720px;object-fit:contain;border:1px solid #e5e7eb}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin-top:12px;color:#526178}@media(max-width:700px){.visuals,.answers{grid-template-columns:1fr}.toolbar{flex-wrap:wrap}main{padding:20px 12px}}
</style><main><div class='eyebrow'>ADAPTIVEVISION-RL / DATA PILOT</div><h1>20 条区域标注试验</h1>
<p>Smart-Train 先导样本 · DeepSeek Flash · 原图回答与局部框</p>
<div class='note'>首轮依据 answer_from_image 是否正确筛选，不要求裁剪图独立可回答。Codex 的目视复核与严格答案匹配分别记录，并非人工真值。最终评测计划使用 Smart-Val 500 条；本页不包含最终评测数据。样本来自 HF 部分转换结果，并非全量随机样本。</div>
<div class='metrics'>""" + metrics + """</div><div class='toolbar'><input id='query' placeholder='搜索问题、答案或样本编号' aria-label='搜索样本'><select id='filter' aria-label='按状态筛选'><option value='all'>全部状态</option><option value='pass'>答案匹配</option><option value='review'>答案待核对</option><option value='error'>错误</option><option value='pending'>待标注</option></select></div>""" + "\n".join(cards) + """</main><script>
const query=document.querySelector('#query'),filter=document.querySelector('#filter');
function update(){for(const card of document.querySelectorAll('.card'))card.hidden=!(card.textContent.toLowerCase().includes(query.value.toLowerCase())&&(filter.value==='all'||card.dataset.status===filter.value));}query.addEventListener('input',update);filter.addEventListener('change',update);
</script><dialog id='zoom'><button id='close-zoom' aria-label='关闭图片'>关闭</button><img id='zoom-image' alt='放大的原图或裁剪'></dialog><style>dialog{max-width:96vw;max-height:96vh;border:0;border-radius:12px;padding:16px}dialog::backdrop{background:#172033bb}#zoom-image{display:block;max-height:85vh;max-width:90vw}#close-zoom{display:block;margin:0 0 12px auto;padding:8px 18px;cursor:pointer}</style><script>
const dialog=document.querySelector('#zoom');for(const a of document.querySelectorAll('a[href^="data:image"]'))a.addEventListener('click',e=>{e.preventDefault();document.querySelector('#zoom-image').src=a.href;dialog.showModal();});document.querySelector('#close-zoom').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
</script></html>"""
    page = page.replace("20 条区域标注试验</h1>", f"20 条区域标注试验 · {html.escape(run_name)}</h1>")
    (report_dir / "index.html").write_text(page)
    usage = dict(tokens)
    markdown = f"""# 20 条区域标注先导试验 · {run_name}

- 样本数：{len(records)}；有效标注：{stats['valid_annotations']}；错误：{stats['errors']}；待标注：{stats['pending']}。
- 答案自动匹配：{stats['answers_matched']}；按答案正确口径接受且有框：{stats['region_reward_eligible']}。
- Codex 目视复核：{stats['visually_reviewed']} 条；复核后答案正确：{stats['answers_correct_after_review']} 条。复核不是独立模型评测或人工真值。
- 定位诊断：目标与必要上下文均可见 {stats['localization_useful_context_region']} 条、上下文不足 {stats['localization_incomplete_context']} 条、漏掉目标细节 {stats['localization_misses_target_detail']} 条、目标文字被截断 {stats['localization_partial_target']} 条、整图框 {stats['localization_whole_image']} 条、答案错误不适用 {stats['localization_not_applicable']} 条。诊断不改变按答案正确筛选的口径。
- 服务／模型：{', '.join(sorted(providers)) or '尚无成功响应'} / {', '.join(sorted(models)) or '尚无成功响应'}。
- API 报告用量：`{json.dumps(usage)}`。不把认证失败算成成功标注。
- 接受依据：严格匹配或独立记录的 Codex 目视复核；原始答案、框和 API 返回全部保留，不通过重试挑选正确答案。
- 不发送标准答案给教师，不做裁剪图独立回答验证；必要上下文要求以本次保存的 `prompt.txt` 为准。
- 框是伪标签；回答正确并不证明定位准确。本报告不宣称人工通过率；逐条复核见 `visual_review.jsonl`。

## 数据范围

本轮是 HF 数据浏览器部分转换数据的诊断性样本。8 个窗口共 800 条候选，固定 seed=20260921，
原 `use_tool` 标记各取 10 条；在抽样前按窗口轮转，并在样本内检查文档 ID、相同图片与 dHash 近重复。
原始字节、处理后图片、来源行号、采样清单及哈希均保存在 `{output}`。
尚未完成全量原始数据扫描、文档标识语义确认或 Train/Val 跨集合去重，也未构造正式 3000/300/500 划分。
正式方案：Smart-Train → 3000 训练 + 300 开发；Smart-Val → 500 最终评测。

## 逐条结果

| 样本 | 状态 | 标准答案 | 教师回答 | 框面积占比 |
|---|---|---|---|---|
""" + "\n".join(table) + "\n"
    (report_dir / "RESULTS.md").write_text(markdown)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/pilot20")
    args = parser.parse_args()
    build_report(args.output.resolve(), args.report_dir.resolve())

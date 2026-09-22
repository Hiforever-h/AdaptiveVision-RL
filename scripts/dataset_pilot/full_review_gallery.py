"""Render all annotations and build an interactive box-editing review gallery."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .common import read_jsonl, write_json


STYLE = r"""
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:#f4f6fa;color:#172033;font:15px/1.5 system-ui,-apple-system,sans-serif}
button,input,select,textarea{font:inherit}.app{display:grid;grid-template-columns:310px minmax(0,1fr);height:100%}.side{background:#111827;color:#e5e7eb;display:flex;flex-direction:column;min-height:0}.side-head{padding:18px;border-bottom:1px solid #334155}.side h1{font-size:18px;margin:0 0 5px}.progress{color:#a5b4fc;font-size:13px}.filters{display:grid;gap:8px;padding:12px;border-bottom:1px solid #334155}.filters input,.filters select{width:100%;border:1px solid #475569;border-radius:7px;background:#1f2937;color:#f8fafc;padding:8px}.list{overflow:auto;padding:7px}.item{display:block;width:100%;border:0;border-left:4px solid transparent;background:transparent;color:#cbd5e1;text-align:left;padding:8px 9px;margin:1px 0;border-radius:5px;cursor:pointer}.item:hover{background:#1f2937}.item.active{background:#312e81;color:white;border-left-color:#a5b4fc}.item.reviewed::after{content:' ✓';color:#86efac}.item.edited::before{content:'✎ ';color:#fbbf24}.empty{padding:20px;color:#94a3b8}.main{overflow:auto;padding:20px 26px 40px}.topbar{display:flex;justify-content:space-between;gap:18px;align-items:center;position:sticky;top:0;background:#f4f6faf2;backdrop-filter:blur(8px);padding:4px 0 14px;z-index:2}.nav,.actions{display:flex;gap:8px;align-items:center}.nav button,.actions button,.box-controls button{border:1px solid #cbd5e1;background:white;border-radius:8px;padding:8px 13px;cursor:pointer}.nav button:disabled{opacity:.4;cursor:default}.counter{font-weight:700}.card{background:white;border:1px solid #dde3ed;border-radius:13px;padding:20px;box-shadow:0 6px 22px #1720330b}.sample-title{display:flex;justify-content:space-between;gap:15px;align-items:start}.sample-title h2{margin:0;font-size:22px}.badges{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}.badge{padding:3px 9px;border-radius:999px;background:#e9eef7;color:#45546b;font-size:12px}.badge.good{background:#dcfce7;color:#166534}.badge.warn{background:#fef3c7;color:#92400e}.question{font-size:18px;white-space:pre-wrap;margin:19px 0}.answers{display:grid;grid-template-columns:1fr 1fr;gap:14px}.answer{background:#f7f8fb;border-radius:9px;padding:12px 14px;overflow-wrap:anywhere}.answer small{display:block;color:#64748b;margin-bottom:4px}.visuals{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,2fr);gap:18px;margin-top:20px;align-items:start}.visual{margin:0;background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:10px;min-width:0}.visual figcaption{color:#64748b;font-size:13px;margin-bottom:8px}.visual canvas{display:block;width:100%;max-height:72vh;object-fit:contain;background:#eef2f7;touch-action:none}.model-link{float:right}.no-crop{height:250px;display:grid;place-items:center;color:#64748b}.box-controls{display:flex;gap:8px;flex-wrap:wrap;align-items:end;margin:15px 0;padding:12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px}.box-controls label{font-size:12px;color:#64748b}.box-controls input,.box-controls select{display:block;width:105px;margin-top:4px;border:1px solid #cbd5e1;border-radius:7px;padding:7px;background:white}.box-controls select{width:125px}.coords{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;color:#dbeafe;border-radius:8px;padding:11px;font:12px/1.5 ui-monospace,monospace}.review{margin-top:20px;border-top:1px solid #e2e8f0;padding-top:18px}.review-grid{display:grid;grid-template-columns:220px 1fr auto;gap:10px;align-items:start}.review select,.review textarea{border:1px solid #cbd5e1;border-radius:8px;padding:9px;background:white;width:100%}.review textarea{min-height:76px;resize:vertical}.save-state{color:#64748b;padding:9px;white-space:nowrap}.hint{color:#64748b;font-size:13px;margin:10px 0 0}@media(max-width:900px){.app{grid-template-columns:1fr}.side{height:330px}.main{padding:14px}.visuals,.answers,.review-grid{grid-template-columns:1fr}.topbar{position:static;flex-wrap:wrap}.badges{justify-content:flex-start}.sample-title{display:block}}
"""


APP_JS = r"""
const records=window.TRAIN_REVIEW_RECORDS;
const recordById=new Map(records.map(r=>[r.sample_id,r]));
const storageKey=window.TRAIN_REVIEW_STORAGE_KEY;
const $=s=>document.querySelector(s);
let reviews={};try{reviews=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){reviews={}}
let filtered=[],currentId=null,draftStatus=null,draftBox=null,drag=null,imageGeneration=0;
const list=$('#list'),search=$('#search'),statusFilter=$('#status-filter'),toolFilter=$('#tool-filter'),reviewFilter=$('#review-filter');
const canvas=$('#editor-canvas'),ctx=canvas.getContext('2d'),cropCanvas=$('#crop-canvas'),cropCtx=cropCanvas.getContext('2d'),editorImage=new Image();

function isReviewed(id){return Boolean(reviews[id]?.verdict)}
function isEdited(id){return Boolean(reviews[id]?.edited)}
function persist(){localStorage.setItem(storageKey,JSON.stringify(reviews));updateProgress()}
function updateProgress(){const reviewed=records.filter(r=>isReviewed(r.sample_id)).length,edited=records.filter(r=>isEdited(r.sample_id)).length;$('#progress').textContent=`已复核 ${reviewed}/${records.length} · 已改框 ${edited} · 未复核 ${records.length-reviewed}`}
function matchesReview(r){const f=reviewFilter.value,v=reviews[r.sample_id]?.verdict||'';if(f==='all')return true;if(f==='unreviewed')return !v;if(f==='reviewed')return !!v;if(f==='edited')return isEdited(r.sample_id);if(f==='pass')return v==='pass';return v&&v!=='pass'}
function applyFilters(){const q=search.value.trim().toLowerCase();filtered=records.filter(r=>(statusFilter.value==='all'||r.status===statusFilter.value)&&(toolFilter.value==='all'||String(r.use_tool)===toolFilter.value)&&matchesReview(r)&&(!q||[r.sample_id,r.question,r.answers.join(' '),r.answer_from_image].join(' ').toLowerCase().includes(q)));renderList();if(!filtered.some(r=>r.sample_id===currentId)&&filtered.length)show(filtered[0].sample_id);if(!filtered.length){list.innerHTML='<div class="empty">没有符合条件的样本</div>';$('#counter').textContent='0 / 0'}}
function renderList(){list.textContent='';const frag=document.createDocumentFragment();for(const r of filtered){const b=document.createElement('button');b.className='item'+(r.sample_id===currentId?' active':'')+(isReviewed(r.sample_id)?' reviewed':'')+(isEdited(r.sample_id)?' edited':'');b.textContent=`${r.sample_id} · ${r.status}`;b.onclick=()=>show(r.sample_id);frag.appendChild(b)}list.appendChild(frag)}
function badge(text,kind=''){return `<span class="badge ${kind}">${text}</span>`}
function currentRecord(){return recordById.get(currentId)}
function sameBox(a,b){return JSON.stringify(a||null)===JSON.stringify(b||null)}
function show(id){const r=recordById.get(id);if(!r)return;currentId=id;const i=filtered.findIndex(x=>x.sample_id===id);$('#counter').textContent=i>=0?`${i+1} / ${filtered.length}`:'—';$('#prev').disabled=i<=0;$('#next').disabled=i<0||i>=filtered.length-1;$('#sample-id').textContent=r.sample_id;$('#badges').innerHTML=badge(r.status,r.status==='localized'?'good':'')+badge(`use_tool=${r.use_tool}`)+badge(r.strict_answer_match?'严格答案匹配':'严格答案不匹配',r.strict_answer_match?'good':'warn');$('#question').textContent=r.question;$('#gold').textContent=r.answers.join(' / ');$('#teacher').textContent=r.answer_from_image;$('#overlay-link').href=r.overlay;$('#model-crop-link').hidden=!r.model_crop;if(r.model_crop)$('#model-crop-link').href=r.model_crop;const review=reviews[id]||{};draftStatus=review.edited_status||r.status;draftBox=Object.prototype.hasOwnProperty.call(review,'edited_reference_box_pixels')?(review.edited_reference_box_pixels?[...review.edited_reference_box_pixels]:null):(r.boxes_pixels[0]?[...r.boxes_pixels[0]]:null);$('#edited-status').value=draftStatus;$('#verdict').value=review.verdict||'';$('#note').value=review.note||'';$('#save-state').textContent=review.updated_at?'已保存':'';syncInputs(false);const generation=++imageGeneration;editorImage.onload=()=>{if(generation!==imageGeneration)return;canvas.width=r.editor_size[0];canvas.height=r.editor_size[1];drawEditor()};editorImage.src=r.editor;renderList();document.querySelector('.item.active')?.scrollIntoView({block:'nearest'})}
function scaleBox(box){const r=currentRecord(),sx=canvas.width/r.width,sy=canvas.height/r.height;return box?[box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy]:null}
function unscalePoint(x,y){const r=currentRecord();return[Math.round(x*r.width/canvas.width),Math.round(y*r.height/canvas.height)]}
function normalizedBox(){const r=currentRecord();return draftBox?draftBox.map((v,i)=>v/(i%2?r.height:r.width)):null}
function updateCoords(){const r=currentRecord();$('#coords').textContent=JSON.stringify({image_size:[r.width,r.height],model_status:r.status,model_box_pixels:r.boxes_pixels[0]||null,edited_status:draftStatus,edited_box_pixels:draftBox,edited_box_normalized:normalizedBox()},null,2)}
function drawEditor(){if(!editorImage.complete||!canvas.width)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(editorImage,0,0,canvas.width,canvas.height);const b=scaleBox(draftBox);if(b){ctx.lineWidth=Math.max(6,canvas.width/180);ctx.strokeStyle='white';ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.lineWidth=Math.max(3,canvas.width/300);ctx.strokeStyle='#ef233c';ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.fillStyle='#ef233c';const handle=Math.max(10,canvas.width/120);for(const[x,y]of[[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]])ctx.fillRect(x-handle/2,y-handle/2,handle,handle)}drawCrop();updateCoords()}
function drawCrop(){const has=draftStatus==='localized'&&draftBox&&editorImage.complete;cropCanvas.hidden=!has;$('#no-crop').hidden=has;if(!has)return;const b=scaleBox(draftBox),w=Math.max(1,Math.round(b[2]-b[0])),h=Math.max(1,Math.round(b[3]-b[1]));cropCanvas.width=w;cropCanvas.height=h;cropCtx.drawImage(editorImage,b[0],b[1],w,h,0,0,w,h)}
function syncInputs(redraw=true){for(const[i,id]of['#x1','#y1','#x2','#y2'].entries()){$(id).value=draftBox?draftBox[i]:'';$(id).disabled=draftStatus!=='localized'}if(redraw)drawEditor();else updateCoords()}
function clampBox(box){const r=currentRecord();let[x1,y1,x2,y2]=box.map(Math.round);x1=Math.max(0,Math.min(r.width-1,x1));y1=Math.max(0,Math.min(r.height-1,y1));x2=Math.max(x1+1,Math.min(r.width,x2));y2=Math.max(y1+1,Math.min(r.height,y2));return[x1,y1,x2,y2]}
function saveReview(){const r=currentRecord(),modelBox=r.boxes_pixels[0]||null,edited=draftStatus!==r.status||!sameBox(draftBox,modelBox),value={verdict:$('#verdict').value,note:$('#note').value.trim(),updated_at:new Date().toISOString()};if(edited)Object.assign(value,{edited:true,original_status:r.status,original_reference_box_pixels:modelBox,edited_status:draftStatus,edited_reference_box_pixels:draftBox});if(!value.verdict&&!value.note&&!edited)delete reviews[currentId];else reviews[currentId]=value;persist();$('#save-state').textContent=reviews[currentId]?'已保存':'';renderList()}
function move(delta){const i=filtered.findIndex(r=>r.sample_id===currentId),next=filtered[i+delta];if(next)show(next.sample_id)}
function pointerPosition(e){const rect=canvas.getBoundingClientRect();return[(e.clientX-rect.left)*canvas.width/rect.width,(e.clientY-rect.top)*canvas.height/rect.height]}

canvas.onpointerdown=e=>{const[x,y]=pointerPosition(e),b=scaleBox(draftBox),radius=Math.max(18,canvas.width/80);if(draftStatus!=='localized'||!b){draftStatus='localized';$('#edited-status').value='localized';const p=unscalePoint(x,y);draftBox=[p[0],p[1],p[0]+1,p[1]+1];drag={mode:'draw',start:p};canvas.setPointerCapture(e.pointerId);return}const corners=[[b[0],b[1],'nw'],[b[2],b[1],'ne'],[b[0],b[3],'sw'],[b[2],b[3],'se']],hit=corners.find(c=>Math.hypot(x-c[0],y-c[1])<=radius);if(hit)drag={mode:hit[2],start:[x,y],box:[...draftBox]};else if(x>=b[0]&&x<=b[2]&&y>=b[1]&&y<=b[3])drag={mode:'move',start:[x,y],box:[...draftBox]};else{const p=unscalePoint(x,y);draftBox=[p[0],p[1],p[0]+1,p[1]+1];drag={mode:'draw',start:p}}canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;const[x,y]=pointerPosition(e),p=unscalePoint(x,y),r=currentRecord();if(drag.mode==='draw'){draftBox=clampBox([Math.min(drag.start[0],p[0]),Math.min(drag.start[1],p[1]),Math.max(drag.start[0]+1,p[0]),Math.max(drag.start[1]+1,p[1])])}else if(drag.mode==='move'){const dx=Math.round((x-drag.start[0])*r.width/canvas.width),dy=Math.round((y-drag.start[1])*r.height/canvas.height),w=drag.box[2]-drag.box[0],h=drag.box[3]-drag.box[1],x1=Math.max(0,Math.min(r.width-w,drag.box[0]+dx)),y1=Math.max(0,Math.min(r.height-h,drag.box[1]+dy));draftBox=[x1,y1,x1+w,y1+h]}else{let b=[...drag.box];if(drag.mode.includes('n'))b[1]=p[1];if(drag.mode.includes('s'))b[3]=p[1];if(drag.mode.includes('w'))b[0]=p[0];if(drag.mode.includes('e'))b[2]=p[0];draftBox=clampBox([Math.min(b[0],b[2]-1),Math.min(b[1],b[3]-1),Math.max(b[0]+1,b[2]),Math.max(b[1]+1,b[3])])}syncInputs()};
canvas.onpointerup=()=>{if(drag){drag=null;saveReview()}};

$('#prev').onclick=()=>move(-1);$('#next').onclick=()=>move(1);for(const element of[search,statusFilter,toolFilter,reviewFilter])element.addEventListener(element===search?'input':'change',applyFilters);$('#verdict').onchange=saveReview;let noteTimer;$('#note').oninput=()=>{clearTimeout(noteTimer);noteTimer=setTimeout(saveReview,350)};
$('#edited-status').onchange=()=>{draftStatus=$('#edited-status').value;const r=currentRecord();if(draftStatus==='localized'&&!draftBox)draftBox=[Math.round(r.width*.2),Math.round(r.height*.2),Math.round(r.width*.8),Math.round(r.height*.8)];if(draftStatus!=='localized')draftBox=null;syncInputs();saveReview()};
for(const[i,id]of['#x1','#y1','#x2','#y2'].entries())$(id).onchange=()=>{const values=['#x1','#y1','#x2','#y2'].map(selector=>Number($(selector).value));if(values.every(Number.isFinite)){draftBox=clampBox(values);syncInputs();saveReview()}};
$('#reset-box').onclick=()=>{const r=currentRecord();draftStatus=r.status;draftBox=r.boxes_pixels[0]?[...r.boxes_pixels[0]]:null;$('#edited-status').value=draftStatus;syncInputs();saveReview()};
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);const map={'1':'pass','2':'answer_wrong','3':'box_wrong','4':'both_wrong','5':'uncertain'};if(map[e.key]){$('#verdict').value=map[e.key];saveReview()}});
$('#export').onclick=()=>{const lines=records.filter(r=>reviews[r.sample_id]).map(r=>JSON.stringify({sample_id:r.sample_id,...reviews[r.sample_id]})).join('\n')+'\n';const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines],{type:'application/jsonl'}));a.download='train3000_visual_review_and_box_edits.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
updateProgress();applyFilters();if(records.length)show(records[0].sample_id);
"""


INDEX = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AdaptiveVision-RL · Train 3,000 全量框复核</title><link rel="stylesheet" href="style.css"></head><body><div class="app">
<aside class="side"><div class="side-head"><h1>Train 3,000 全量框复核</h1><div id="progress" class="progress"></div></div><div class="filters"><input id="search" placeholder="搜索 ID、问题或答案"><select id="status-filter"><option value="all">全部定位状态</option><option value="localized">localized</option><option value="global">global</option><option value="uncertain">uncertain</option></select><select id="tool-filter"><option value="all">全部 use_tool</option><option value="true">use_tool=true</option><option value="false">use_tool=false</option></select><select id="review-filter"><option value="all">全部复核状态</option><option value="unreviewed">未复核</option><option value="reviewed">已复核</option><option value="edited">已修改框</option><option value="pass">仅通过</option><option value="problem">仅有问题</option></select></div><div id="list" class="list"></div></aside>
<main class="main"><div class="topbar"><div class="nav"><button id="prev">← 上一条</button><button id="next">下一条 →</button><span id="counter" class="counter"></span></div><div class="actions"><button id="export">导出复核与框修改 JSONL</button></div></div><section class="card"><div class="sample-title"><h2 id="sample-id"></h2><div id="badges" class="badges"></div></div><div id="question" class="question"></div><div class="answers"><div class="answer"><small>标准答案</small><strong id="gold"></strong></div><div class="answer"><small>GLM answer_from_image</small><strong id="teacher"></strong></div></div>
<div class="visuals"><figure class="visual"><figcaption>框编辑器：拖动框、拖四角缩放，或在空白处重新绘制 <a id="overlay-link" class="model-link" target="_blank">模型原框图</a></figcaption><canvas id="editor-canvas"></canvas></figure><figure class="visual"><figcaption>修改后框实时裁剪预览 <a id="model-crop-link" class="model-link" target="_blank">模型原裁剪</a></figcaption><canvas id="crop-canvas"></canvas><div id="no-crop" class="no-crop">当前状态没有局部框</div></figure></div>
<div class="box-controls"><label>状态<select id="edited-status"><option value="localized">localized</option><option value="global">global</option><option value="uncertain">uncertain</option></select></label><label>x1<input id="x1" type="number" min="0"></label><label>y1<input id="y1" type="number" min="0"></label><label>x2<input id="x2" type="number" min="1"></label><label>y2<input id="y2" type="number" min="1"></label><button id="reset-box">恢复模型原框</button></div><details><summary>坐标详情</summary><pre id="coords" class="coords"></pre></details>
<div class="review"><div class="review-grid"><select id="verdict"><option value="">未复核</option><option value="pass">通过</option><option value="answer_wrong">答案错误</option><option value="box_wrong">框有问题</option><option value="both_wrong">答案和框都有问题</option><option value="uncertain">需进一步复查</option></select><textarea id="note" placeholder="复核备注，可选"></textarea><span id="save-state" class="save-state"></span></div><p class="hint">方向键切换样本；1=通过，2=答案错误，3=框有问题，4=答案和框都有问题，5=需进一步复查。框修改、结论与备注保存在当前浏览器；导出的 JSONL 同时包含模型原框和修改后框。</p></div></section></main></div><script src="records.js"></script><script src="app.js"></script></body></html>"""


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_atomic(image: Image.Image, path: Path, image_format: str, **kwargs) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format=image_format, **kwargs)
    temporary.replace(path)


def render_record(source: Path, report_dir: Path, row: dict) -> dict:
    sample_id = row["sample_id"]
    annotation = row["region_annotation"]
    boxes = annotation.get("reference_boxes_pixels", [])
    overlay_relative = Path("assets/overlays") / f"{sample_id}.png"
    crop_relative = Path("assets/crops") / f"{sample_id}.png"
    editor_relative = Path("assets/editor") / f"{sample_id}.jpg"
    overlay_path = report_dir / overlay_relative
    crop_path = report_dir / crop_relative
    editor_path = report_dir / editor_relative
    if not overlay_path.exists() or not editor_path.exists() or (boxes and not crop_path.exists()):
        original = Image.open(source / row["image_path"]).convert("RGB")
        if not overlay_path.exists():
            overlay = original.copy()
            draw = ImageDraw.Draw(overlay)
            line = max(3, round(min(original.size) / 180))
            for box in boxes:
                rectangle = (box[0], box[1], box[2] - 1, box[3] - 1)
                draw.rectangle(rectangle, outline="#ffffff", width=line + 4)
                draw.rectangle(rectangle, outline="#ef233c", width=line)
            save_atomic(overlay, overlay_path, "PNG")
        if boxes and not crop_path.exists():
            save_atomic(original.crop(boxes[0]), crop_path, "PNG")
        if not editor_path.exists():
            editor = ImageOps.contain(original, (1800, 1800), Image.Resampling.LANCZOS)
            save_atomic(editor, editor_path, "JPEG", quality=94, subsampling=0)
    with Image.open(editor_path) as editor:
        editor_size = list(editor.size)
    quality = row["quality_checks"]
    return {
        "sample_id": sample_id, "question": row["question"], "answers": row["answers"],
        "answer_from_image": annotation["answer_from_image"], "status": annotation["status"],
        "strict_answer_match": bool(quality["answer_match"]["match"]),
        "use_tool": bool(row["source_info"].get("use_tool")),
        "width": row["width"], "height": row["height"],
        "boxes_pixels": boxes, "boxes_normalized": annotation.get("reference_boxes", []),
        "overlay": overlay_relative.as_posix(),
        "model_crop": crop_relative.as_posix() if boxes else None,
        "editor": editor_relative.as_posix(), "editor_size": editor_size,
    }


def build(source: Path, report_dir: Path, workers: int) -> dict:
    manifest_path = source / "manifest.jsonl"
    annotation_path = source / "annotations.jsonl"
    manifest = read_jsonl(manifest_path)
    annotations = {row["sample_id"]: row for row in read_jsonl(annotation_path)}
    if len(manifest) != len(annotations):
        raise ValueError("Manifest and annotation counts differ")
    records = []
    for row in manifest:
        annotation = annotations.get(row["sample_id"])
        if not annotation or annotation.get("annotation_error"):
            raise ValueError(f"Missing or invalid annotation: {row['sample_id']}")
        records.append(annotation)
    manifest_sha = file_digest(manifest_path)
    annotation_sha = file_digest(annotation_path)
    config = {"source": str(source), "manifest_sha256": manifest_sha,
              "annotations_sha256": annotation_sha, "sample_count": len(records)}
    report_dir.mkdir(parents=True, exist_ok=True)
    config_path = report_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Report directory belongs to a different manifest or annotation file")
    write_json(config_path, config)
    for directory in ("assets/overlays", "assets/crops", "assets/editor"):
        (report_dir / directory).mkdir(parents=True, exist_ok=True)

    rendered = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(render_record, source, report_dir, row): row["sample_id"]
                   for row in records}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            rendered[result["sample_id"]] = result
            if completed % 100 == 0 or completed == len(futures):
                print(f"rendered {completed}/{len(futures)}", flush=True)
    ordered = [rendered[row["sample_id"]] for row in records]
    counts = Counter(row["status"] for row in ordered)
    summary = {
        **config, "created_at": datetime.now(timezone.utc).isoformat(),
        "localized": counts["localized"], "global": counts["global"],
        "uncertain": counts["uncertain"],
        "strict_answer_match": sum(row["strict_answer_match"] for row in ordered),
        "overlay_count": len(list((report_dir / "assets/overlays").glob("*.png"))),
        "model_crop_count": len(list((report_dir / "assets/crops").glob("*.png"))),
        "editor_preview_count": len(list((report_dir / "assets/editor").glob("*.jpg"))),
        "review_state": "browser localStorage; export reviews and pixel-box edits from index.html as JSONL",
    }
    write_json(report_dir / "summary.json", summary)
    storage_key = f"adaptivevision-train-review-{annotation_sha[:16]}"
    (report_dir / "style.css").write_text(STYLE)
    (report_dir / "app.js").write_text(APP_JS)
    records_js = ("window.TRAIN_REVIEW_STORAGE_KEY=" + json.dumps(storage_key) + ";\n"
                  "window.TRAIN_REVIEW_RECORDS="
                  + json.dumps(ordered, ensure_ascii=False, separators=(",", ":")) + ";\n")
    (report_dir / "records.js").write_text(records_js)
    (report_dir / "index.html").write_text(INDEX)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    build(args.source.resolve(), args.report_dir.resolve(), args.workers)

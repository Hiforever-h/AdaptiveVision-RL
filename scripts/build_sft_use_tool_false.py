"""Assemble manually authored SFT labels for source use_tool=false rows.

This script only assembles authored labels, copies answers[0] verbatim, converts
reviewed source boxes when route=tool, creates the corresponding crop, and
validates the result. It does not generate reasoning text or route decisions.
"""
from __future__ import annotations
import json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adaptive_vision_rl.prompts import INITIAL_PROMPT, SECOND_PROMPT
from adaptive_vision_rl.protocol import parse_action
SOURCE=ROOT/'data/visionthink_3000_300_500_balanced/train/annotations.jsonl'
OUT=ROOT/'data/sft_adaptive_vision_v1/use_tool_false'
OUT_REL=Path('data/sft_adaptive_vision_v1/use_tool_false')
DATASET_REL=Path('data/visionthink_3000_300_500_balanced')
def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def write_jsonl(path, rows):
    with path.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
def qbox(box,w,h):
    x1,y1,x2,y2=box
    return [(1000*x1)//w,(1000*y1)//h,(1000*x2+w-1)//w,(1000*y2+h-1)//h]
def crop_box(box,w,h):
    x1,y1,x2,y2=box
    return [math.floor(x1*w/1000),math.floor(y1*h/1000),math.ceil(x2*w/1000),math.ceil(y2*h/1000)]
def tagged(think,tag,body): return f'<think>{think}</think>\n\n<{tag}>{body}</{tag}>'
def build():
    OUT.mkdir(parents=True,exist_ok=True)
    source=read_jsonl(SOURCE)
    selected=[r for r in source if r.get('source_info',{}).get('use_tool') is False]
    if len(selected)!=1500: raise ValueError(f'Expected 1500 source rows, got {len(selected)}')
    labels=read_jsonl(OUT/'manual_labels.jsonl')
    by_id={}
    for label in labels:
        sid=label.get('sample_id')
        if not sid or sid in by_id: raise ValueError(f'Missing/duplicate manual label: {sid}')
        if label.get('route') not in {'direct','tool'}: raise ValueError(f'Invalid route: {label}')
        if not label.get('think_first','').strip(): raise ValueError(f'Missing first think: {sid}')
        if label['route']=='tool' and not label.get('think_second','').strip(): raise ValueError(f'Missing second think: {sid}')
        by_id[sid]=label
    selected_ids={r['sample_id'] for r in selected}
    if set(by_id)!=selected_ids:
        missing=selected_ids-set(by_id); extra=set(by_id)-selected_ids
        raise ValueError(f'Manual labels incomplete: {len(missing)} missing ({sorted(missing)[:5]}), {len(extra)} extra')
    anns=[]; flat=[]; second=[]; counts={'tool':0,'direct':0}; now=datetime.now(timezone.utc).isoformat()
    for src in selected:
        sid=src['sample_id']; label=by_id[sid]; route=label['route']; q=src['question']
        answers=src.get('answers') or []
        if not answers or not isinstance(answers[0],str) or not answers[0]: raise ValueError(f'Missing gold: {sid}')
        gold=answers[0]; lw,lh=src['lowres_size']; ow,oh=src['width'],src['height']
        low=(DATASET_REL/'train'/src['lowres_path']).as_posix()
        original=(DATASET_REL/'train'/src['image_path']).as_posix()
        prompt=INITIAL_PROMPT.format(question=q,width=lw,height=lh)
        qbox_value=original_box=executed=crop_rel=None
        if route=='direct':
            target=tagged(label['think_first'],'answer',gold)
        else:
            region=src.get('region_annotation') or {}
            boxes=region.get('reference_boxes_pixels') or []
            if not boxes: raise ValueError(f'No source crop box for tool route: {sid}')
            original_box=[int(v) for v in boxes[0]]
            qbox_value=qbox(original_box,ow,oh); executed=crop_box(qbox_value,ow,oh)
            if not (0<=qbox_value[0]<qbox_value[2]<=1000 and 0<=qbox_value[1]<qbox_value[3]<=1000): raise ValueError(f'Invalid Qwen box: {sid} {qbox_value}')
            crop_rel=(OUT_REL/'crops'/f'{sid}.png').as_posix(); dest=ROOT/crop_rel; dest.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(ROOT/original) as im:
                im=im.convert('RGB')
                if im.size!=(ow,oh): raise ValueError(f'Original size mismatch: {sid}')
                crop=im.crop(tuple(executed))
                if not crop.width or not crop.height: raise ValueError(f'Empty crop: {sid}')
                crop.save(dest,format='PNG',optimize=True)
            tc=json.dumps({'name':'request_local_region','arguments':{'bbox_2d':qbox_value}},separators=(',',':'))
            target=tagged(label['think_first'],'tool_call',tc)
        parsed=parse_action(target,allow_tool=(route=='tool'),image_size=(lw,lh))
        expected='tool' if route=='tool' else 'answer'
        if not parsed.valid or parsed.kind!=expected: raise ValueError(f'Invalid first action {sid}: {parsed.error}')
        if route=='direct' and parsed.answer!=gold: raise ValueError(f'Direct gold mismatch: {sid}')
        turn0={'turn_index':0,'stage':'decision','prompt_template':'INITIAL_PROMPT','prompt':prompt,'input_images':[low],'assistant':target}
        turns=[turn0]
        ann={'sample_id':sid,'source_split':'train','question':q,'reference_answers':answers,'lowres_path':low,'original_image_path_for_provenance':original,'lowres_size':[lw,lh],'original_size':[ow,oh],'source_use_tool_hint':False,'source_region_status':(src.get('region_annotation') or {}).get('status','uncertain'),'route':route,'crop_path':crop_rel,'gold_bbox_original_xyxy':original_box,'tool_bbox_qwen_1000_xyxy':qbox_value,'crop_bbox_executed_original_xyxy':executed,'turns':turns,'annotation_meta':{'annotator':'Codex','method':'Per-sample authored route and English think; answer copied verbatim from source answers[0].','think_language':'English','think_word_limit':None,'created_at':now}}
        flat.append({'sample_id':sid,'turn_index':0,'stage':'decision','prompt_template':'INITIAL_PROMPT','prompt':prompt,'input_images':[low],'target':target,'route':route,'crop_path':crop_rel,'answer_target_source':'source answers[0] (verbatim)' if route=='direct' else None})
        counts[route]+=1
        if route=='tool':
            prompt2=SECOND_PROMPT.format(question=q)
            target2=tagged(label['think_second'],'answer',gold)
            parsed2=parse_action(target2,allow_tool=False,image_size=(lw,lh))
            if not parsed2.valid or parsed2.kind!='answer' or parsed2.answer!=gold: raise ValueError(f'Invalid second target / gold mismatch: {sid}')
            t2={'turn_index':1,'stage':'answer_after_tool','prompt_template':'SECOND_PROMPT','prompt':prompt2,'input_images':[low,crop_rel],'assistant':target2,'answer_target_source':'source annotations.jsonl: answers[0] (verbatim)','think_author':'Codex, individually authored'}
            ann['turns'].append(t2); flat.append({'sample_id':sid,'turn_index':1,'stage':'answer_after_tool','prompt_template':'SECOND_PROMPT','prompt':prompt2,'input_images':[low,crop_rel],'target':target2,'route':'tool','crop_path':crop_rel,'answer_target_source':'source annotations.jsonl: answers[0] (verbatim)'}); second.append(flat[-1])
        anns.append(ann)
    write_jsonl(OUT/'annotations.jsonl',anns); write_jsonl(OUT/'turns.jsonl',flat); write_jsonl(OUT/'second_turns.jsonl',second)
    manifest={'source':'data/visionthink_3000_300_500_balanced/train/annotations.jsonl','selection':'source_info.use_tool == false','sample_count':len(anns),'direct_count':counts['direct'],'tool_count':counts['tool'],'turn_count':len(flat),'second_turn_count':len(second),'crop_count':counts['tool'],'coordinate_space':'bbox_2d is Qwen 0-1000 normalized xyxy on the full image; source pixel boxes converted from original image dimensions. Crops replay Qwen coordinates against original images using environment floor/ceil conversion.','answer_policy':'Every <answer> copied verbatim from source answers[0].','think_policy':'English per-sample authored by Codex; no fixed word limit; labels read from manual_labels.jsonl.','created_at':now}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('# use_tool=false SFT labels\n\nThis directory contains the 1,500 source training records with `source_info.use_tool == false`, manually labeled with direct-answer or one-crop trajectories. Answers are copied verbatim from source `answers[0]`.\n\n- `annotations.jsonl`: one complete trajectory per source sample.\n- `turns.jsonl`: flattened turns.\n- `second_turns.jsonl`: answer-after-crop turns for tool routes.\n- `manual_labels.jsonl`: manually authored English route/think labels consumed by the assembler.\n- `crops/`: generated high-resolution crops for tool routes only.\n\nRebuild with the project Python runtime: `python scripts/build_sft_use_tool_false.py`. The script assembles labels and validates the exact gold-answer strings; it does not generate route or reasoning labels.\n',encoding='utf-8')
    print(json.dumps({'samples':len(anns),'direct':counts['direct'],'tool':counts['tool'],'turns':len(flat),'crops':counts['tool'],'manual_labels':len(labels)},ensure_ascii=False))
if __name__=='__main__': build()

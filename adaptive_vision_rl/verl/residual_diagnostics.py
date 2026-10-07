"""Fixed-response comparisons after old/update micro-batches are aligned."""

from __future__ import annotations

import math

import torch


def actor_phases():
    # Keep layout, entropy calculation and update mode fixed within each pair.
    return [
        ("actor_old_micro2", False, False),
        ("actor_update_micro2", True, False),
        ("actor_update_micro2_repeat", True, False),
        ("actor_old_micro2_no_reduced_bf16", False, True),
        ("actor_update_micro2_no_reduced_bf16", True, True),
        ("actor_update_micro2_no_reduced_bf16_repeat", True, True),
        ("actor_old_micro2_after_control", False, False),
    ]


def saved_micro2(records, width):
    values = torch.zeros((len(records), width), dtype=torch.float32)
    for row, record in enumerate(records):
        saved = record["log_probs"].get("actor_old_micro2")
        count = len(record["response_token_ids"])
        if saved is None or len(saved) != count or count > width or any(v is None or not math.isfinite(v) for v in saved):
            raise ValueError("Residual diagnostics require complete actor_old_micro2 data from the source report")
        values[row, :count] = torch.tensor(saved)
    return values


def comparisons(values, mask, *, clip_low, clip_high):
    from adaptive_vision_rl.verl.consistency import probability_difference
    from adaptive_vision_rl.verl.update_diagnostics import ratio_statistics

    pairs = [
        ("saved_actor_old_micro2", "actor_old_micro2"),
        ("actor_old_micro2", "actor_update_micro2"),
        ("actor_update_micro2", "actor_update_micro2_repeat"),
        ("actor_old_micro2", "actor_old_micro2_after_control"),
        ("actor_old_micro2", "actor_old_micro2_no_reduced_bf16"),
        ("actor_old_micro2_no_reduced_bf16", "actor_update_micro2_no_reduced_bf16"),
        ("actor_update_micro2_no_reduced_bf16", "actor_update_micro2_no_reduced_bf16_repeat"),
        ("vllm_prefill", "vllm_prefill_same_wake_repeat"),
        ("vllm_prefill", "vllm_prefill_after_wake"),
        ("rollout", "vllm_prefill"),
    ]
    for vllm in ("rollout", "vllm_prefill", "vllm_prefill_same_wake_repeat", "vllm_prefill_after_wake"):
        pairs.extend((vllm, actor) for actor in ("actor_old_micro2", "actor_old_micro2_no_reduced_bf16"))
    diffs = {f"{left}_vs_{right}": probability_difference(values[left], values[right], mask)
             for left, right in pairs if left in values and right in values}
    ratio_pairs = [("actor_old_micro2", "actor_update_micro2"),
                   ("actor_old_micro2_no_reduced_bf16", "actor_update_micro2_no_reduced_bf16")]
    ratios = {f"{old}_vs_{new}": ratio_statistics(values[old], values[new], mask,
                                                 clip_low=clip_low, clip_high=clip_high)
              for old, new in ratio_pairs if old in values and new in values}
    return diffs, ratios


def focus_probabilities(records, *, limit=4):
    """Rank current residuals, rather than the earlier micro=4 anomalies."""
    tokens = []
    for row, record in enumerate(records):
        values = record["log_probs"]
        actor = values.get("actor_old_micro2")
        if actor is None:
            continue
        for offset, token_id in enumerate(record["response_token_ids"]):
            probabilities = {name: math.exp(seq[offset]) if seq[offset] is not None else None
                             for name, seq in values.items()}
            left, right = probabilities.get("rollout"), probabilities.get("actor_old_micro2")
            if left is not None and right is not None:
                tokens.append(dict(row=row, sample_id=record["anchor"]["sample_id"], response_offset=offset,
                                   token_id=token_id, probability_difference=abs(left-right), probabilities=probabilities))
    return sorted(tokens, key=lambda item: item["probability_difference"], reverse=True)[:limit]


def markdown(report):
    lines = ["# micro=2 生成端残余概率差诊断", "", f"状态：`{report['status']}`", "",
             "固定原报告的样本、图片、token 和 position_ids，加载相同 step 300 权重。不启动训练，不执行 backward/optimizer，不保存 checkpoint。",
             "rollout 为之前真实生成时保存的概率；本轮不重新采样。vLLM 将固定回答拼回输入，只使用其 prompt_logprobs，额外生成的一个 token 被丢弃。",
             "actor 的旧概率重算和真实更新模式前向都固定 micro=2。唯一精度控制是在 actor 中临时禁止 BF16 reduced-precision reduction；vLLM 保持原设置。", "",
             f"actor LoRA 匹配 checkpoint：`{report.get('checkpoint_lora_matches_actor', '未完成')}`；"
             f"前向前后 LoRA 未改变：`{report.get('actor_weights_unchanged', '未完成')}`。",
             f"源 checkpoint 文件未改变：`{report.get('checkpoint_file_unchanged', '未完成')}`；"
             f"精度设置已恢复：`{report.get('precision_controls_restored', '未完成')}`。", "",
             "vllm_prefill 与 same_wake_repeat 使用同一次唤醒和同一 LoRA ID；after_wake 在另一次唤醒及重新同步后重算。每次请求都清空前缀缓存，逐样本运行。",
             "先检查 vLLM 自身重复差，再看保存的 rollout→prefill 和 prefill→actor。它们是对照关系，绝对差不能直接相加分解。", ""]
    for case in report["cases"]:
        lines.extend([f"## {case['name']}", "", "| 对照 | mean | std | max | 非有限 token |",
                      "|---|---:|---:|---:|---:|"])
        for name, item in case["comparisons"].items():
            numbers = [f"{item[key]:.6f}" if key in item else "n/a" for key in ("mean", "std", "max")]
            lines.append(f"| {name} | {' | '.join(numbers)} | {item['nonfinite_tokens']} |")
        for name, item in case.get("update_ratios", {}).items():
            lines.extend(["", f"{name}：无更新 ratio 范围 `{item.get('min')}`～`{item.get('max')}`，"
                          f"超界比例 `{item.get('fraction_outside_bounds')}`；该比例不是 pg_clipfrac。"])
        for phase, evidence in case.get("update_forward_evidence", {}).items():
            lines.extend(["", f"{phase}：train/grad/checkpointing 证据有效 `{evidence.get('valid', False)}`，"
                          f"模式恢复 `{evidence.get('training_modes_restored', False)}`。"])
        lines.extend(["", "当前概率差最大的 token（全部阶段概率见 report.json）：", ""])
        for token in case.get("residual_focus_tokens", []):
            p = token["probabilities"]
            lines.append(f"- {token['sample_id']} / offset {token['response_offset']} / token {token['token_id']}："
                         f"rollout={p.get('rollout')}，actor={p.get('actor_old_micro2')}，prefill={p.get('vllm_prefill')}。")
        for phase in case["errors"]:
            lines.extend(["", f"失败阶段：{phase}；堆栈见 report.json。"])
    lines.extend(["", "## LoRA 同步与退出", "",
                  "| 同步 | 已检查/应检查矩阵 | GPU 槽位一致 |", "|---|---:|---|"])
    for event in report.get("sync_events", []):
        slots = event.get("gpu_slots", {})
        lines.append(f"| {event['event']} | {slots.get('checked_matrices', '?')}/{slots.get('expected_matrices', '?')} | {slots.get('all_match', False)} |")
    lines.extend(["", f"显式资源清理：`{report.get('shutdown', {}).get('status', '未开始')}`。",
                  "外层 process_exit.json 单独记录子进程退出码；completed 仅表示数值检查完整完成，不能证明训练稳定或忽略进程崩溃。",
                  "mixed_turns 的 vLLM 数据复用前两组逐样本重算结果，仅 actor 重新以单双图交错分组前向。",
                  "某个控制改善部分 token，不能直接证明唯一根因或将它作为正式修复。", ""])
    if report.get("fatal_error"):
        lines.extend(["```text", report["fatal_error"], "```", ""])
    return "\n".join(lines)

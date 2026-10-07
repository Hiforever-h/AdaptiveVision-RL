# DTPO 模型评测

训练前检查 Thinking 基座的原始输出格式，见 [Thinking 基座输出格式检查](THINKING_FORMAT_PROBE.md)。

评测脚本对冻结的 500 条 Test 数据执行确定性两轮推理。它复用训练时的提示词、
动作解析、裁剪坐标换算、答案精确准确率与数值相似度部分奖励、格式奖励和 Coverage+IoU 区域奖励口径，
并直接用 vLLM 加载 DTPO checkpoint 保存的 LoRA adapter，无需先合并模型。动态 LoRA
评测会将 vLLM V1 EngineCore 留在当前进程，使 Qwen3-VL 模块映射补丁在模型加载
和多模态 profiling 时都生效。

SFT、训练 rollout 与独立评测共用 `adaptive_vision_rl/images.py`。推理批处理传递
原始图片，由 `fit_image_prompt` 在生成前执行一次尺寸归一化；加载图片时统一应用
EXIF 方向，使低清图与高清裁剪处于同一个坐标系。训练、Val/Test 和基准评测都只对
协议合法的最终答案给答案分，缺少有效 `<think>`、标签外有多余文本或再次调用工具
的最终输出均记为答错。

## 运行环境

在训练使用的 Linux CUDA 环境和项目根目录下运行。默认的训练输出根目录和
Test 数据路径分别为：

```text
/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_lora
data/visionthink_3000_300_500_balanced/test/annotations.jsonl
```

训练 checkpoint 自动将当前 adapter 保存到 `global_step_N/actor/lora_adapter`。
评测脚本的 `--checkpoint` 参数可以指向以下目录：

- 训练输出根目录；
- `global_step_N`；
- `global_step_N/actor`；
- `global_step_N/actor/lora_adapter`。

传入训练输出根目录时，脚本优先读取
`latest_checkpointed_iteration.txt`，否则选择编号最大的完整 adapter。
adapter 必须同时包含 `adapter_config.json` 和 `adapter_model.safetensors`。
也可直接传入独立 adapter 目录。旧 checkpoint 若只有 `.pt`，可先用
`python -m scripts.export_dtpo_lora --checkpoint /path/to/global_step_N --output /path/to/adapter`
离线导出一次，再将输出目录传给 `--checkpoint`。

## 先做小规模检查

```bash
python scripts/evaluate_dtpo.py \
  --checkpoint /root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_lora \
  --limit 8 \
  --output-dir /root/autodl-tmp/outputs/evaluation/dtpo_smoke
```

检查成功后运行完整 Test 500：

```bash
python scripts/evaluate_dtpo.py \
  --checkpoint /root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_lora \
  --output-dir /root/autodl-tmp/outputs/evaluation/qwen3vl_4b_dtpo_lora_test
```

重新写入已有目录时需要显式添加 `--overwrite`。如果实际 checkpoint 不在默认位置：

```bash
python scripts/evaluate_dtpo.py \
  --checkpoint /path/to/qwen3vl_4b_dtpo_lora \
  --output-dir /path/to/evaluation-output
```

显存不足时可降低 `--batch-size` 或 `--gpu-memory-utilization`。`--batch-size` 只控制
每批送入脚本的样本数；vLLM 仍会在批内动态调度。默认 greedy decoding 参数与训练
validation 保持一致：temperature 0、单条输出、最多 1024 个 response token。
默认基座为 `/root/autodl-tmp/models/qwen3vl_4b_sft_final_merged`，与 DTPO
训练配置一致。评估与训练使用同一模板适配，提示词只预填 assistant 前缀，
由 completion 生成完整 `<think>...</think>`。如果训练时覆盖了
`actor_rollout_ref.model.path`，评测时也须通过 `--model` 指定同一个基座。

## 输出

输出目录包含：

- `predictions.jsonl`：逐样本两轮原始输出、解析结果、最终答案、正确性、裁剪框、
  区域奖励、视觉 token 和摊销生成时间；
- `summary.json`：模型与 adapter 指纹、数据文件指纹、解码参数、总体指标和按原始
  `use_tool` 提示分层的指标。

主要指标包括精确准确率、平均答案分、直接回答/工具路径准确率、工具调用率、格式合规率、无效动作率、
合格工具调用的几何奖励、获取及实际处理的视觉 token、相对全图 token 比例和吞吐。
区域框是离线伪标签，因此几何奖励用于诊断工具行为，不应代替最终答案准确率。

这里的 `answer_score` 是原始答案相似度，`outcome_reward` 是
`answer_score + format_reward` 的诊断值。训练中的 DTPO Outcome Reward 还会将
非精确答案分乘以 `1 - balance_penalty`，并按轨迹组扣除 balance cost。
旧评测产物不会自动更新；预处理与格式口径修复后，应重新运行评测。奖励逻辑修复
只影响后续训练，已有权重需要重新训练才能消除此前奖励反转的影响。

`estimated_generation_seconds_per_sample` 是批处理生成时间按该轮样本数摊销后的估计值；
吞吐以实际整段墙钟时间计算，它不是单请求在线服务的延迟测量。

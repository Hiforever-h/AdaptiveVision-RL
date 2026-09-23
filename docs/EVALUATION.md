# DTPO 模型评测

评测脚本对冻结的 500 条 Test 数据执行确定性两轮推理。它复用训练时的提示词、
动作解析、裁剪坐标换算、答案精确匹配、格式奖励和 Coverage+IoU 区域奖励口径，
并直接用 vLLM 加载 verl-agent 保存的 LoRA adapter，无需先合并模型。

## 运行环境

在训练使用的 Linux CUDA 环境和项目根目录下运行。脚本默认读取：

```text
/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_lora
data/visionthink_3000_300_500_balanced/test/annotations.jsonl
```

checkpoint 参数可以指向以下任一层级：

- 训练输出根目录；
- `global_step_N`；
- `global_step_N/actor`；
- `global_step_N/actor/lora_adapter`。

传入训练输出根目录时，脚本优先读取 `latest_checkpointed_iteration.txt`，否则选择
编号最大的完整 checkpoint。adapter 必须同时包含 `adapter_config.json` 和
`adapter_model.safetensors`。

## 先做小规模检查

```bash
python scripts/evaluate_dtpo.py \
  --limit 8 \
  --output-dir /root/autodl-tmp/outputs/evaluation/dtpo_smoke
```

检查成功后运行完整 Test 500：

```bash
python scripts/evaluate_dtpo.py \
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
validation 保持一致：temperature 0、单条输出、最多 512 个 response token。

## 输出

输出目录包含：

- `predictions.jsonl`：逐样本两轮原始输出、解析结果、最终答案、正确性、裁剪框、
  区域奖励、视觉 token 和摊销生成时间；
- `summary.json`：模型与 adapter 指纹、数据文件指纹、解码参数、总体指标和按原始
  `use_tool` 提示分层的指标。

主要指标包括准确率、直接回答/工具路径准确率、工具调用率、格式合规率、无效动作率、
合格工具调用的几何奖励、获取及实际处理的视觉 token、相对全图 token 比例和吞吐。
区域框是离线伪标签，因此几何奖励用于诊断工具行为，不应代替最终答案准确率。

`estimated_generation_seconds_per_sample` 是批处理生成时间按该轮样本数摊销后的估计值；
吞吐以实际整段墙钟时间计算，它不是单请求在线服务的延迟测量。

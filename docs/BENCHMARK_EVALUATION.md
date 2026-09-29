# 六个数据集评测

评测输入是 `data/eval_6bench_3000_v1/`：ChartQA、OCRBench、MME、RealWorldQA、POPE、MathVerse 各 500 道题，总计 3000 道。MMVet 已从这套评测集中排除。`scripts/evaluate_benchmarks.py` 参考了 `sft/evaluate.py` 实际调用的 `scripts/evaluate_dtpo.py`，复用其 vLLM 推理、LoRA 加载、裁图和图像 token 计数，并复用训练时的低清首轮及局部高清第二轮 prompt。

在**Linux CUDA** 机器上、项目 Python 环境中，从仓库根目录运行。`--base-model`、`--mode` 和 `--output-dir` 均必填。`--output-dir` 指定本次结果的输出目录。每次运行只评测 `low_tool` 或 `high_only` 一种模式；不传 `--adapter` 时仅评测 base model，传入时仅评测 base model 加该 adapter 后的模型。

```bash
python scripts/evaluate_benchmarks.py \
  --base-model /path/to/base-model \
  --mode low_tool \
  --output-dir outputs/benchmark_base_low
```

```bash
python scripts/evaluate_benchmarks.py \
  --base-model /path/to/base-model \
  --adapter /path/to/lora-adapter \
  --mode high_only \
  --output-dir outputs/benchmark_lora_high
```

`--adapter` 可以指向直接含有 `adapter_config.json` 和 `adapter_model.safetensors` 的目录，也可以指向脚本支持的 verl checkpoint 根目录。`--batch-size` 默认 8。显存不足时可减小它或调整 `--gpu-memory-utilization`、`--max-model-len`。对要求远程代码的模型可加 `--trust-remote-code`。

先做少量题的流程检查：

```bash
python scripts/evaluate_benchmarks.py \
  --base-model /path/to/base-model \
  --mode low_tool \
  --limit 12 \
  --output-dir outputs/benchmark_smoke
```

`--limit` 按六个数据集轮流取题，故上例每个数据集两题。结果会标记 `partial: true`；正式评测去掉 `--limit`。已有输出默认拒绝覆盖，确认要重跑时加 `--overwrite`。建议每个模型或 adapter 使用独立输出目录，避免混淆。

## 两种模式

- `low_tool`：第一轮给原问题和低清全图。模型可直接给 `<answer>`，也可请求一次 `request_local_region`。如果请求有效，脚本从原始高清图裁出对应区域，第二轮给原问题、低清全图及局部高清图，必须给 `<answer>`。脚本不会发起第三轮。工具坐标是低清全图上 0–1000 归一化的 `xyxy`。
- `high_only`：给原问题和高清全图，只生成一轮，必须直接给 `<answer>`。任何工具调用都判为格式无效，不执行工具。

两种模式都要求先有非空 `<think>...</think>`，再有且仅有一个 `<answer>...</answer>` 或首轮允许的 `<tool_call>...</tool_call>`。格式无效或第二轮没有合法答案时，该题准确率记为 0，逐题文件记录解析错误。模型输出使用贪心解码（temperature 0）。

## 输出与指标

输出目录包含 `summary.json` 和本次模式对应的 `predictions_low_tool.jsonl` 或 `predictions_high_only.jsonl`。`summary.json` 的 `metrics.<mode>.overall` 是**按题数加权**的汇总准确率；`metrics.<mode>.by_dataset.<数据集名>` 给每个数据集的独立指标。`macro_accuracy` 是六个数据集准确率的简单平均；完整评测各 500 题，因此与总准确率相同。`partial` 运行时两者可能不同。

每个指标块都含 `count`、`correct_count`、`accuracy`、`direct_answer_accuracy`、`tool_answer_accuracy`、`tool_call_rate`、`format_compliance`、首轮及最终无效比例，以及图像 token 均值。`high_only` 的工具调用率恒为 0，因为工具请求不执行；若模型输出工具调用，会体现在格式无效与错误答案中。

两个图像比例均按每道题计算，再取平均，分母是单次高清全图的图像 token 数：

```text
vision_ratio      = (首轮低清全图 + 局部高清图) / 单次高清全图
vision_ratio_true = (首轮低清全图 + 第二轮低清全图 + 局部高清图) / 单次高清全图
```

直接在第一轮回答时，局部高清图及第二轮低清图都计 0；`high_only` 两项均为 1。分母按 `high_only` prompt 对高清全图进行同样的尺寸适配后计数，分子使用实际生成时的图像 token 计数。因此比例是**模型视觉 token 开销**，不是源图像像素面积比例。输入图像仍受 `scripts/evaluate_dtpo.py` 沿用的单图最大像素和模型上下文限制。

## 本地判分规则

本脚本不调用外部判分 API。通用规则是 Unicode/大小写/空白规范化后的精确匹配，纯数字支持数值相等；Yes/No 题按此规则处理。ChartQA 的纯数字答案额外采用相对参考值 5% 的容差。OCRBench 允许规范化后的参考文字出现在模型短答案中。MathVerse 选择题接受选项字母或 `option B` 等简短形式；自由填答使用本地规范化精确匹配和纯数字相等。后者**不是 MathVerse 官方语言模型判分**，语义等价但文字不同的答案可能被低估；需要与官方分数对比时，需另行使用官方判分流程。

评测集合及抽样来源详见 [BENCHMARK_DATA.md](BENCHMARK_DATA.md)。可先运行 `python scripts/sample_benchmark_eval.py verify` 校验 3000 道题及图像配对。

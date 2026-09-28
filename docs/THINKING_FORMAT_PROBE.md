# Thinking 基座输出格式检查

在 A800 的 Linux CUDA 环境、项目根目录和已安装 `requirements.txt` 的 Python 环境中运行：

```bash
python scripts/probe_thinking_format.py \
  --val-parquet data/verl_agent/val.parquet \
  --dataset-root data/visionthink_3000_300_500_balanced \
  --limit 32 \
  --batch-size 4 \
  --reference-second-probes 8 \
  --output-dir /root/autodl-tmp/outputs/thinking_format_probe
```

脚本直接加载 `Qwen/Qwen3-VL-4B-Thinking` 基座，不加载 LoRA。默认使用
temperature 0 和每轮最多 1024 个生成 token。它从 Val parquet 的 `env_kwargs`
读取真实问题和图片路径；该文件的 `prompt` 列只是训练框架所需的占位文本。
脚本为 vLLM 0.11.0 的 EngineCore 设置 `spawn` 启动方式，避免 CUDA 已在
主进程初始化后再被 `fork`；无效的 `OMP_NUM_THREADS` 值会改为 `1`。

首轮使用低清全图。模型给出合法工具调用时，脚本按其框裁剪原图，继续生成真实第二轮。
若工具调用不足，脚本还会为最多 8 条带参考框的样本构造裁剪图，单独检查第二轮的
`<think>...</think><answer>...</answer>` 格式。这类 `reference_second_turn`
只用于格式检查，不代表模型自行选择了工具或正确定位。

结果写入输出目录：

- `completions.jsonl`：每条样本的原始 completion、首个生成 token ID、结束原因、生成 token 数、
  `<think>` 起止检查、解析结果，以及真实或参考裁剪下的第二轮输出。
- `summary.json`：首轮、真实第二轮和参考裁剪第二轮分别统计的标签率、合法动作率、
  长度截断率和解析错误分布，并记录 `<think>` 的 tokenizer ID。

重点查看 `first_turn.starts_with_think_rate` 和 `valid_action_rate`。
如果 `natural_second_turn.count` 为 0，先看首轮原始输出是否缺少 `<think>`、
工具 JSON 是否无效，或生成是否因长度截断；仍可通过
`reference_second_turn` 检查第二轮格式。复测可换一个 `--output-dir`，
或显式添加 `--overwrite`。之后可用 `--temperature 1.0` 再检查训练采样条件下的格式稳定性。

若要重点检查工具调用，可用 `--source-use-tool true --limit 32` 选择原数据中
带工具提示的 Val 样本，并换一个输出目录。`source_use_tool` 只是原数据的弱提示，
不能当作“必须调用工具”的正确标签。汇总中会分别列出这一提示为 true 和 false
时的首轮格式与动作统计；参考框构造的第二轮也优先选提示为 true 的样本。

当独立评测与 DTPO 初始验证的工具调用率不一致时，可在同一模型和 Val 切片上比较
`--prompt-mode text` 与 `--prompt-mode ids`。前者使用独立评测的字符串 prompt 输入；
后者像 DTPO collector 一样，用 `add_special_tokens=False` 把相同的渲染后 prompt
编码成 `prompt_token_ids`。两次运行应指定不同的 `--output-dir`，并设
`--reference-second-probes 0` 只比较首轮动作。两种模式都未启用 DTPO actor 的 LoRA，
因此相同结果只能排除 prompt 输入形式，不能排除 actor/rollout 权重同步问题。

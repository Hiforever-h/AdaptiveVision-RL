# Qwen3-VL-4B-Thinking 的 LoRA SFT

本目录使用已经构造好的 2,250 个 turn 训练一个 epoch：750 个直接回答决策、750 个调用工具的决策，以及 750 个拿到裁剪图后的回答。训练输入与项目现有的单图、双图推理格式一致。损失只计算助手输出的 `<think>`、动作和结束标记；先对每个 turn 的输出 token 求平均，再对一个 micro batch 中的 turn 求平均。

## 目录与路径

默认配置见 `sft/config.json`：

| 用途 | 默认值 | 覆盖方式 |
| --- | --- | --- |
| 基座模型 | `Qwen/Qwen3-VL-4B-Thinking` | `--model`：Hub 模型 ID 或完整的本地模型目录 |
| 数据目录 | 项目根目录下的 `data/` | `--data-dir` |
| SFT JSONL | 数据目录下的 `sft_adaptive_vision_v1/train_2250_turns/turns.jsonl` | `--turns`：绝对路径或相对于数据目录的路径 |
| LoRA checkpoint 输出 | `/root/autodl-tmp/checkpoints/qwen3vl_4b_sft_lora` | `--output-dir` |
| 验证、测试数据 | 数据目录下的 `visionthink_3000_300_500_balanced/` | `--dataset-root` |
| 评测结果 | LoRA 输出目录下的 `evaluation/` | `--eval-output-dir`（一键脚本）或评测脚本的 `--output-dir` |

`--data-dir` 应指向**包含** `sft_adaptive_vision_v1/` 和 `visionthink_3000_300_500_balanced/` 的目录。例如，把整个 `data/` 复制到 `/root/autodl-tmp/data/` 后，传入 `--data-dir /root/autodl-tmp/data`。JSONL 中以 `data/` 开头的图像路径会映射到这个目录；不要把 `--data-dir` 设成 `data/sft_adaptive_vision_v1/`。

## Hugging Face 缓存与镜像

可以在**启动 Python 前**设置：

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
```

脚本通过 Transformers 加载模型和处理器，这些环境变量由 `huggingface_hub` 自动读取，无需写入配置文件。`HF_HOME` 指定 Hugging Face 主目录，`HF_HUB_CACHE` 指定下载的基座模型缓存位置，`HF_ENDPOINT` 指定 Hub 请求地址。请在运行脚本的同一个 shell 中先执行 `export`；若模型已经放在本地完整目录，也可以用 `--model /绝对路径/模型目录` 直接加载。镜像地址能否成功下载仍取决于镜像服务本身。

**基座模型权重**会缓存到 `HF_HUB_CACHE`；**训练生成的 LoRA adapter** 会写到 `--output-dir`。默认两者都位于 `/root/autodl-tmp/` 下。这里保存的是 LoRA 权重，不会再复制一份完整基座模型。

## 在单张 A800 80G 上运行

先安装项目 `requirements.txt` 的依赖，并按项目说明单独安装 `flash-attn==2.7.4.post1`。从项目根目录执行：

```bash
# 可选：先用实际模型处理器检查全部 2,250 个 turn 的长度和图像输入。
python -m sft.train --preflight-only --data-dir /root/autodl-tmp/data

# 训练，然后评测 Val300 两个 checkpoint 与选中版本的 Test500。
bash sft/run_sft.sh \
  --model Qwen/Qwen3-VL-4B-Thinking \
  --data-dir /root/autodl-tmp/data \
  --output-dir /root/autodl-tmp/checkpoints/qwen3vl_4b_sft_lora
```

如果数据仍在项目根目录下，省略 `--data-dir` 即可。如果不用一键脚本，也可以分别执行：

```bash
python -m sft.train \
  --model /root/autodl-tmp/models/Qwen3-VL-4B-Thinking \
  --data-dir /root/autodl-tmp/data \
  --output-dir /root/autodl-tmp/checkpoints/my_sft

python -m sft.evaluate \
  --model /root/autodl-tmp/models/Qwen3-VL-4B-Thinking \
  --data-dir /root/autodl-tmp/data \
  --checkpoint-dir /root/autodl-tmp/checkpoints/my_sft \
  --test
```

如需使用另一份训练 JSONL，给训练脚本或一键脚本增加 `--turns /绝对路径/turns.jsonl`。如果 Val/Test 单独存放，再增加 `--dataset-root /绝对路径/visionthink_3000_300_500_balanced`。

## 训练与评测规则

默认超参为 LoRA rank 64、alpha 128、dropout 0.05；仅适配语言模型的投影层。训练使用 BF16、FlashAttention 2、梯度检查点、AdamW、学习率 `5e-5`、余弦调度、3% warmup、权重衰减 0.01。micro batch 为 2、梯度累积为 8，有效 batch 为 16。输入不截断：prompt 上限 6,144 token，助手输出（含 `<|im_end|>`）上限 1,024 token。训练前会使用实际模型处理器检查全部样本。

预计共 141 个优化器 step，分别在 `checkpoint-70/` 和 `final/` 保存 LoRA adapter。`sft/evaluate.py` 在现有 Val300 上分别运行完整的两轮推理，只用**最终答案正确率**选择 checkpoint；若正确率相同，选择半轮版本。随后只用选中的 adapter 评测现有 Test500。选择结果记录在 `evaluation/selection.json`。底层项目评测器也会保存详细诊断文件，但这些指标不参与选择。

如果训练已完成、只需重跑评测，执行 `python -m sft.evaluate --test --overwrite`，并按需附上相同的 `--model`、`--data-dir`、`--checkpoint-dir`。如果首次训练遇到显存不足，可把 `sft/config.json` 中的 micro batch 改为 1、梯度累积改为 16，保持有效 batch 为 16。中断后可从半轮 checkpoint 恢复：`python -m sft.train --resume-from-checkpoint /root/autodl-tmp/checkpoints/qwen3vl_4b_sft_lora/checkpoint-70 --skip-preflight`，并传入原先使用的路径参数。

## 合并并导出完整模型

先查看 `evaluation/selection.json` 的 `selected_adapter`，然后把该目录传给合并脚本。例如选中整轮 checkpoint 时：

```bash
python -m sft.merge_lora \
  --base-model Qwen/Qwen3-VL-4B-Thinking \
  --adapter /root/autodl-tmp/checkpoints/qwen3vl_4b_sft_lora/final \
  --output-dir /root/autodl-tmp/models/qwen3vl_4b_sft_merged
```

如果选中半轮 checkpoint，就把 `--adapter` 改为相同训练输出目录下的 `checkpoint-70`。`--base-model` 可换成训练时所用基座模型的完整本地目录，必须使用与训练时相同的基座权重。合并脚本默认在 CUDA 上以 BF16 加载，使用安全合并，然后导出完整的 safetensors 模型分片、配置文件和处理器；也可指定 `--dtype`、`--device cpu` 或 `--max-shard-size 5GB`。输出目录必须是尚不存在的新路径，不会覆盖已有文件。`merge_manifest.json` 记录输入来源与版本。

合并后的目录可直接作为完整模型加载。导出的处理器保留基座模型原始 chat template；推理时应像项目评测器一样先调用 `configure_thinking_tokenizer`，让模型自行生成 `<think>`。原始 LoRA adapter 仍保留在训练输出目录。后续 GRPO 若要从 SFT 权重继续训练，还需单独配置项目锁定版本的 verl-agent actor。

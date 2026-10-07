# Step 300 训推一致性验证

在服务器的项目根目录、原训练 CUDA 环境中运行。默认读取：

```text
/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_run2/global_step_300
```

同步本次代码到服务器后，在空闲的单卡 A800 上执行：

```bash
conda activate bisight-rl
CUDA_VISIBLE_DEVICES=0 python -m scripts.verify_rollout_consistency
```

脚本自动使用项目内 `third_party/verl-agent` checkout，并检查其提交是否符合
`third_party/verl-agent.commit`。使用真实的 `ActorRolloutRefWorker`、FSDP 包装、
LoRA 在线同步、trajectory collector 和图像处理流程；不启动训练循环或 WandB。
它只加载 `.pt` 模型权重，不恢复优化器、不更新参数、不导出 adapter、不旋转 checkpoint。
checkpoint 中无需存在 `adapter/` 或 `actor/lora_adapter/`，也无需额外复制完整模型或 checkpoint。
加载后直接从 FSDP 展开的 LoRA 层读取 A/B 权重，避免新版 PEFT 使用模块路径筛选时，
嵌套 FSDP 包装路径与权重字典路径不一致而漏掉 adapter；首次和后续在线同步、
checkpoint 导出共用此收集器。

默认从 `data/verl_agent/val.parquet` 取前 4 个样本，保留训练配置中的 1024 response token
上限。每个样本分别检查单图决策首轮和固定工具裁剪后的双图第二轮；第二轮不是模型
自然调用工具的结果，而是保证双图分支一定得到检查的控制实验。参考框缺失时使用中心框。
随后将这两组已采样结果交错混合到同一个 actor micro-batch，再检查一次，覆盖训练中
单图与双图、长短序列混合的情况；不会为此重新采样 response。

如果 run2 使用了不同的基座目录、LoRA 配置或图像数据路径，传入原训练值，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.verify_rollout_consistency \
  --checkpoint /root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_run2/global_step_300 \
  --base-model /root/autodl-tmp/models/qwen3vl_4b_sft_final_merged \
  --data-root data/visionthink_3000_300_500_balanced \
  --override actor_rollout_ref.model.lora_rank=64 \
  --override actor_rollout_ref.model.lora_alpha=128
```

`.pt` 中没有独立的 LoRA alpha 元数据，脚本无法推断它；默认 rank 64、alpha 128
来自项目配置。其他训练配置变化通过重复 `--override KEY=VALUE` 指定；也可用
`--config` 指向原训练 YAML。

输出目录默认为：

```text
/root/autodl-tmp/outputs/rollout_consistency_step300_YYYYMMDD_HHMMSS/
```

结束时会打印同名 `.zip` 文件的完整路径。**下载整个 `.zip` 提供分析即可**，
无需下载 checkpoint。即使运行失败，也会生成含错误堆栈的报告和压缩包；已存在的
输出目录会被拒绝，以免覆盖之前的报告。

## 对照内容

| 对照 | 检查内容 |
|---|---|
| `.pt` LoRA → actor | 全部 A/B 张量的 SHA256 指纹、dtype、数量、非零 B、有限值 |
| actor → vLLM GPU 槽位 | 每次同步核对所有语言层 A/B；包含 packed QKV、gate/up 和 alpha/r 缩放 |
| rollout vs actor_training | 同一 response，按训练的去 padding 开关、micro-batch 和温度重算概率 |
| actor_training vs actor_packed_single | 保持同一 response，对比训练批次与单样本前向 |
| actor_packed_single vs actor_padded_single | 对比去 padding 与普通带 padding 前向 |
| mixed_turns / actor_training vs actor_separate_turns | 同一 response，交错混合单图/双图与按轮次分开重算的差别 |
| rollout vs vllm_prefill | 将已采样 response 拼回原输入，使用 prompt_logprobs 固定 token 重算；检查 decode/prefill 差异 |
| vllm_prefill vs actor | 固定同一 response，比较两个引擎；温度非 1 时额外使用 actor_raw |
| vllm_base_prefill vs actor_base | 两端均关闭 DTPO LoRA，判断差异是否依赖 adapter |
| vllm_prefill vs vllm_base_prefill / actor vs actor_base | 对照两端各自开启/关闭 LoRA 的实际影响 |

vLLM 固定 token 重算每次只处理一个样本，并清空前缀缓存，减少峰值显存和缓存干扰。
诊断将 context 上限至少增加 1 个 token，以容纳固定完整 response 后的一个未使用生成 token。
actor 的重算不额外计算 entropy，不改变 logprob 算法；没有 backward 或 optimizer step。

生成时和固定 token 重算时都会核对 vLLM 实际展开后的 prompt token IDs 与 actor 输入。
如果多模态 token 不一致，脚本会报告首个不一致位置，固定 token 重算结果记为缺失，
不会通过错位取值制造一个看似正常的概率差。

## 报告文件

- `REPORT.md`：主要概率差、首次和后续 LoRA 同步结果、定位说明。
- `report.json`：运行版本、配置、权重检查、汇总、各阶段错误堆栈。
- `decision_samples.json` / `reference_second_turn_samples.json` / `mixed_turns_samples.json`：输入 token、位置编码、
  图像尺寸和 grid、逐 token logprob、生成文本、差异最大的 token 与上下文。
- `effective_config.yaml`：实际使用的配置。
- `run.log`：Python 运行日志。

统计排除 response padding，并记录所有缺失或非有限 token。未完成的阶段和槽位检查
明确标记；不会将缺失结果当成通过。`completed` 只表示诊断阶段全部执行完成，
不表示概率已一致。没有使用任意阈值自动宣布修复。

脚本只读取所选 checkpoint、基座和数据。报告保存 token、输出文本和配置，不打包图像或模型权重。

可以用 `--samples 8 --offset 4` 扩大覆盖；如果只想先检查入口，使用
`--samples 1 --max-response-tokens 128`。后者会记录缩短后的配置，不应替代默认长度的正式对照。
若某个 GPU 阶段报错，先保留并提供对应 `.zip`，无需先反复调整配置。

本地已验证报告计算、token 对齐、失败打包与 CPU 权重检查；完整 GPU 对照需要在服务器执行。

## 固定异常 token 的后续诊断

若已确认同步权重正确，但 batch / 去 padding 对照仍有明显差异，可以重放上一轮报告中的
原始 response。将本次代码同步到服务器，在原 CUDA 环境运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.verify_rollout_consistency \
  --replay-report /root/autodl-tmp/outputs/rollout_consistency_step300_20261007_193610 \
  --forward-diagnostics
```

仍使用默认 step 300 checkpoint。其他 checkpoint 需要显式传入 `--checkpoint`。
`--replay-report` 可以指向报告目录或其中的 `report.json`；目录必须包含原来的
`*_samples.json`。配置来自原报告的 `effective_config`，样本按原 sample ID 从 parquet
重建，不允许同时改变样本范围、response 上限或使用 `--override`。模型/图像路径搬迁
可使用 `--base-model`、`--data-root`，但内容应与原测试一致。原始 report 与 .pt 都不修改，
输出另建带时间戳的目录及 `.zip`。

重放会核对 prompt token、位置编码、图像尺寸/grid 和 LoRA 指纹；回答保持原样，
因此能持续观察同一个异常 `<` token。**`rollout` 概率取自原报告，actor 和
`vllm_prefill` 概率使用当前代码重算，本轮没有重新采样。**

新增以下前向对照，覆盖单图、双图和两者交错的 batch：

| 阶段 | 用途 |
|---|---|
| `actor_training_repeat` / `actor_training_after_controls` | 重复同一前向，并检查临时控制恢复后的结果 |
| `actor_padded_batch` | 与 packed batch、padded single 对照，区分 batch 与去 padding 的影响 |
| `actor_base_*` | 关闭 LoRA 后重复不同布局，判断差异是否依赖 LoRA；均使用温度 1 |
| `actor_fp32_logprob` | 仅将最终 logprob 运算的 logits 转为 FP32 |
| `actor_fp32_head_*` | 临时用 FP32 计算输出层矩阵乘法，其他层沿用原精度 |
| `actor_no_reduced_bf16_*` | 临时禁止 BF16 GEMM 的 reduced-precision reduction |
| `actor_precise_head_*` | 同时使用上述输出层与 GEMM 控制 |

这些是定位用的临时控制，不修改参数存储或正式训练配置。FP32 输出层使用前向时
已物化的权重转成 FP32，并不恢复已在 BF16 转换中丢失的信息。禁止 BF16 reduction
仅影响采用该选项的 CUDA 内核；无改善不能排除所有数值问题。相关行为见
[PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)。

每个场景新增 `*_forward_trace.json`，记录原报告中概率差最大的 4 个 token 的预测位置：
语言层输入、每层输出、输出层输入及完整 logits，以及它们所在样本的视觉 merger /
DeepStack 特征差。`REPORT.md` 同时给出这些 token 在不同控制下的概率。
“最早出现变化的已记录阶段”只代表这些有限的记录点，不能直接当成根因定位。
精度控制改善概率差时仍需结合激活差判断，不能仅凭改善就认定原实现错误。

诊断过程不调用 backward / optimizer，也不写入 checkpoint 或 adapter。结束后提供
新生成的 `.zip`，即可继续区分同步之外的前向差异。

## 真实更新模式 micro-batch=2 与 SFT 10 步短训

同步代码后，在原服务器环境、项目根目录运行：

```bash
conda activate bisight-rl
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_dtpo_micro2_probe \
  --replay-report /root/autodl-tmp/outputs/rollout_consistency_step300_20261007_214712
```

源目录需要包含 `report.json` 和三份 `*_samples.json`。该入口依次启动两个独立进程，
前一阶段退出并释放显存后才开始后一阶段：

1. 只加载 step 300 的模型权重，固定原输入和回答。旧概率使用真实
   `compute_log_prob`（eval/no_grad、计算 entropy），分别按 micro=4 和 micro=2 重算。
   更新前向直接调用真实 `_forward_micro_batch`，使用 train、启用梯度、启用非重入
   gradient checkpointing、micro=2，并重复一次。覆盖单图、双图及交错混合场景。
   不调用 backward/optimizer，不运行原有 18 项精度/布局控制，不重新运行 vLLM prefill。
2. **从 SFT 基座新建 LoRA，关闭自动恢复，训练 step 1–10**。旧概率和更新都使用 micro=2，
   其他模型精度、采样、PPO、DTPO 参数沿用源报告。worker 按完整训练长度初始化
   optimizer/scheduler 后，才把 driver 的停止条件改为 10，因此学习率及 warmup
   与完整新训练的前 10 步一致。关闭 checkpoint 保存、全量验证及 WandB；
   输出逐步本地指标和 rollout 文本。只修改这次短训配置，不修改默认训练 YAML。

旧 step 300 保留用于重现异常，不用于初始化这次短训。当前 LoRA 训练和 checkpoint
导出代码没有自动合并 adapter 或写回 SFT 基座目录的步骤；先前评估的图片处理问题
会影响评估输入和结果，不会因此把错误权重写入 SFT 基座。

前向报告比较 `exp(update_logprob - old_logprob)`：重点看旧概率统一为 2 后，
无参数更新时的 ratio 是否更接近 1，以及超出实际 PPO clip 边界的比例是否下降。
这个超界比例不是训练的 `pg_clipfrac`，后者还依赖 advantage 符号。
重复前向和 train/grad/checkpointing 的实际 hook 证据也会记录。

默认输出 `/root/autodl-tmp/outputs/dtpo_micro2_step10_时间戳/`，结束时打印同名 zip：

- `REPORT.md`：两阶段状态、10 步概率差、PPO clipping/KL、梯度范数、学习率和任务指标。
- `forward/REPORT.md` / `forward/report.json` / `forward/*_samples.json`：前向对照和逐 token 数据。
- `train/metrics.jsonl` / `train/run_state.json`：每步完整指标、实际步数、scheduler horizon 和 warmup。
- `train/rollouts/`：这 10 步的 rollout 文本。
- `short_train_config.json` / `train/effective_config.json`：启动和实际运行配置。
- `forward.log` / `train.log` / `run.json`：日志、源报告 SHA256 与失败堆栈。

下载整个 zip 即可，不包含模型/adapter 权重。失败也会打包已有结果；前向检查未完成或
出现非有限 token 时不会开始短训。`completed` 表示阶段执行完成，不能代替概率差的
人工评估；10 步也不足以证明最终精度或长期稳定性。需要单独运行前向时可使用：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.verify_rollout_consistency \
  --replay-report /root/autodl-tmp/outputs/rollout_consistency_step300_20261007_214712 \
  --update-forward-diagnostics
```

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
无需额外复制完整模型或 checkpoint。

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

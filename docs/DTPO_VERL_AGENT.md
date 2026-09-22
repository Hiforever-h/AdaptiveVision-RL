# 基于 verl-agent 的 DTPO + LoRA 训练

本实现面向单张 A800 80 GB，并固定使用
`third_party/verl-agent.commit` 中记录的 verl-agent 提交。项目不直接包含
verl-agent 源码，而是通过项目侧适配器接入其环境循环、FSDP/vLLM 混合 Worker、
LoRA、rollout 分组、日志和 checkpoint 功能。

## macOS 开发边界

当前 Mac 仅用于：

- 数据转换；
- 协议、奖励和 DTPO 核心逻辑单测；
- Python 与 Shell 语法检查；
- 和固定 verl-agent 提交进行静态接口核对。

CUDA 训练栈不能在当前 Mac 上运行，包括 vLLM、FlashAttention 和 FSDP GPU
Worker。训练启动脚本会在 macOS 上提前退出，避免将平台依赖错误误认为训练代码错误。

Mac 本地检查命令：

```bash
conda env update -f environment.yml
conda activate vison-rl
python scripts/prepare_verl_data.py
python -m unittest discover -s tests -v
```

## 已实现的训练语义

- 第一轮输入低分辨率全图和问题，策略选择直接回答或请求一次局部高清裁剪。
- 合法裁剪请求会产生第二轮模型输出，第二轮不得再次调用工具。
- 第二轮同时输入低分辨率全图和高清局部图；vLLM 已显式设置为每个 prompt
  最多接收两张图片。
- verl-agent 对每个 turn 单独前向，因此：
  - 唯一获取的视觉 token 数为 `low + crop`；
  - 两轮轨迹实际处理的视觉 token 数为 `low + (low + crop)`。
  两种口径都会记录。
- Outcome Reward 为答案正确性、0.5 格式奖励和论文形式的 balance reward 之和。
  Balance penalty 从论文的 0.1 降为 0.01，阈值保持 0.2。
- PPO 使用论文的非对称裁剪范围：下界 0.20、上界 0.24；学习率为
  `1e-6`，不使用 KL，工具优势系数为 0.3。
- Tool Reward 对所有候选参考框取
  `sqrt(Coverage * IoU)` 的最大值。没有合格参考框的样本不计算工具优势。
- 一轮直接回答仅使用 `A_outcome`。两轮轨迹的完整第一轮输出使用
  `A_outcome + 0.3 * A_tool`，第二轮回答使用 `A_outcome`。
- PPO 分别对工具轮 token 和回答轮 token 做归一化。每个训练 step 生成
  32 条轨迹，展开后得到 32～64 个 turn row；不足 64 的部分使用零 loss
  padding 补齐。Padding 不参与奖励、优势、token 分母或训练指标。
- 当前答案正确性采用项目已有的确定性精确匹配／数值匹配，不在训练期间调用在线
  Judge 模型。

## A800 环境要求

建议使用：

- Linux x86_64；
- NVIDIA A800 80 GB；
- 可兼容 vLLM 0.11.0 所需 CUDA/PyTorch Wheel 的 NVIDIA 驱动；
- Conda；
- Python 3.12；
- 至少预留模型、数据、编译缓存和 checkpoint 所需磁盘空间。

根目录的 `requirements.txt` 记录了 Python 依赖。`flash-attn` 没有放入该文件，
因为它必须在 PyTorch 已安装后使用 `--no-build-isolation` 单独编译安装。

## 安装

以下命令均在项目根目录执行。

### 1. 创建独立环境

```bash
conda create -n adaptive-vision-dtpo python=3.12 pip -y
conda activate adaptive-vision-dtpo
python -m pip install --upgrade pip setuptools wheel
```

### 2. 获取固定版本的 verl-agent

```bash
git clone https://github.com/langfengQ/verl-agent.git third_party/verl-agent
git -C third_party/verl-agent checkout "$(cat third_party/verl-agent.commit)"
```

启动脚本会再次校验实际提交是否与 `third_party/verl-agent.commit` 一致。

### 3. 安装普通 Python 与 GPU Runtime 依赖

```bash
python -m pip install -r requirements.txt
```

其中 `vllm==0.11.0` 会解析与其匹配的 PyTorch 依赖。建议使用干净的 Conda
环境，避免服务器预装的 PyTorch/CUDA Python 包造成版本冲突。

项目通过 `adaptive_vision_rl.verl_agent_hooks` 回移了 vLLM 0.11.2 对
Qwen3-VL 多模态模块前缀的修复。该 hook 只在检测到 `vllm==0.11.0` 且原始
错误映射仍存在时生效，使 rank 64、alpha 128 的 LoRA 只挂载到语言模型，
避免视觉塔在 vLLM profiling 阶段错误进入 LoRA 路径。启动日志中应出现：

```text
Applied the vLLM 0.11.2 Qwen3-VL LoRA mapping backport
```

### 4. 单独安装 FlashAttention

```bash
python -m pip install flash-attn==2.7.4.post1 \
  --no-build-isolation \
  --no-cache-dir
```

### 5. 安装固定提交的 verl-agent

依赖已由根目录 `requirements.txt` 安装，因此这里不再让 pip 重新解析依赖：

```bash
python -m pip install -e third_party/verl-agent --no-deps
```

### 6. 登录 WandB

默认配置同时启用终端日志和 WandB。首次使用时执行：

```bash
wandb login
```

也可以通过环境变量提供密钥，并把本地日志放到数据盘：

```bash
export WANDB_API_KEY=<your-key>
export WANDB_DIR=/root/autodl-tmp/wandb
```

密钥不要写入 YAML 或提交到 Git。无网络时可使用
`export WANDB_MODE=offline`，训练结束后再运行 `wandb sync`。

项目不会修改 verl-agent checkout。自定义逻辑仅覆盖：

- 两轮视觉环境和多图 collector；
- Reward 与 Advantage 构造；
- DTPO 工具轮／回答轮 loss 聚合；
- 人工 padding 的零 loss 屏蔽；
- 视觉 token 和 DTPO 指标。

LoRA 只作用于语言模型的 attention 和 MLP 投影层，视觉编码器保持冻结。这也避免
向 vLLM rollout 热加载其不支持的视觉模块 LoRA key。

## 准备训练数据

```bash
python scripts/prepare_verl_data.py
```

该命令从冻结的 JSONL 数据生成：

```text
data/verl_agent/train.parquet
data/verl_agent/val.parquet
data/verl_agent/test.parquet
```

Parquet 文件可重复生成且已被 Git 忽略。图片仍保存在
`data/visionthink_3000_300_500_balanced/`，训练环境按需加载，不把原图写入
Parquet。

## 运行 CPU 检查

在 A800 上安装完整依赖后，环境相关测试不应再因缺少 NumPy 而跳过：

```bash
python -m unittest discover -s tests -v
python -m compileall -q adaptive_vision_rl scripts/prepare_verl_data.py tests
bash -n scripts/run_dtpo_lora.sh
```

## 两步 GPU Smoke Test

正式训练前建议先运行两步训练，检查模型加载、多图 rollout、LoRA 热加载、DTPO
loss 和 checkpoint 路径：

```bash
bash scripts/run_dtpo_lora.sh \
  trainer.total_training_steps=2 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1
```

Smoke test 需要重点确认：

- vLLM 能接收第二轮的两张图片；
- 32～64 个真实 turn row 能正确 padding 到 64；
- `loss_mask` 中 padding 权重为 0；
- LoRA 权重能在 FSDP Actor 和 vLLM rollout 间同步；
- 日志中出现 DTPO 与视觉 token 指标。

## 正式训练

```bash
bash scripts/run_dtpo_lora.sh
```

默认每 250 个训练 step 保存一次可恢复训练的 checkpoint，并只保留最近 1 个：

```yaml
trainer:
  save_freq: 250
  max_actor_ckpt_to_keep: 1
```

最后一个训练 step 无论能否被 250 整除都会保存。输出目录由
`trainer.default_local_dir` 控制。当前默认训练输出均写到 AutoDL 数据盘：

```text
/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_lora  # checkpoint 与最终 LoRA adapter
/root/autodl-tmp/outputs/rollouts/qwen3vl_4b_dtpo_lora  # rollout JSONL
/root/autodl-tmp/wandb  # WandB 本地日志
```

默认 WandB project 为 `adaptive_vision_rl`，run name 为
`qwen3vl_4b_dtpo_lora`。可在启动时覆盖，例如：

```bash
bash scripts/run_dtpo_lora.sh \
  trainer.project_name=adaptive_vision_rl \
  trainer.experiment_name=qwen3vl_4b_dtpo_lora_a800_run1
```

终端会显示带 elapsed、ETA 和 steps/s 的训练进度条。ETA 在完成第一个训练
step 后出现，并使用平滑后的 step 时间持续更新；初始 validation 不计入训练进度。
同一估计还会写入 WandB：

- `progress/completion_percent`；
- `progress/step_time_ema_seconds`；
- `progress/steps_per_hour`；
- `progress/eta_hours`；
- `progress/estimated_total_hours`。

验证和 checkpoint step 通常更慢，因此这些 step 结束后 ETA 会相应调整。若临时不想
上传 WandB，可在启动命令后添加 `'trainer.logger=[console]'`。

可通过 OmegaConf dot-list 参数覆盖配置，例如：

```bash
bash scripts/run_dtpo_lora.sh \
  trainer.total_training_steps=80 \
  trainer.test_freq=20 \
  trainer.save_freq=20
```

默认配置位于 `configs/dtpo_qwen3vl_4b_lora.yaml`。重要日志包括：

- `dtpo/accuracy`；
- `dtpo/direct_answer_accuracy`；
- `dtpo/tool_answer_accuracy`；
- `dtpo/tool_call_rate`；
- `dtpo/tool_reward`；
- `vision/tokens_low`；
- `vision/tokens_crop`；
- `vision/tokens_acquired`；
- `vision/tokens_processed`；
- `vision/token_ratio`。

## Batch 口径

默认设置如下：

| 配置 | 数值 | 含义 |
| --- | ---: | --- |
| `data.train_batch_size` | 4 | 每个训练 step 的不同问题数 |
| `env.rollout.n` | 8 | 每个问题在线采样的轨迹数 |
| 真实 trajectory 数 | 32 | `4 × 8` |
| 真实 turn row 数 | 32～64 | 直接回答 1 行，工具轨迹 2 行 |
| Padding 后 row 数 | 64 | Padding row 的 loss 为 0 |
| `ppo_mini_batch_size` | 64 | 覆盖完整的 step-expanded batch |
| `ppo_micro_batch_size_per_gpu` | 1 | 每次前向／反向处理的 row 数 |
| 梯度累积次数 | 64 | `64 ÷ 1` |
| `rollout.log_prob_micro_batch_size_per_gpu` | 1 | 旧策略 log-prob micro batch |

## 集成约束

`actor_rollout_ref.rollout.multi_turn.enable` 在 YAML 中必须保持为 `false`。
训练入口会在 verl-agent 完成配置校验和 Worker 初始化后，仅打开 Actor 使用
`loss_mask` 的分支；真正的两轮交互由 `AdaptiveVisionTrajectoryCollector` 和自定义
环境管理。

Entropy 和两条 KL 路径必须保持关闭，因为当前 `loss_mask` 存储的是 DTPO 精确
归一化权重，而不只是布尔 mask。当前实现限定单机单卡，不能直接通过增加 GPU 数量
扩展；多卡版本需要重新处理 mini-batch 归一化和跨 rank token 计数。

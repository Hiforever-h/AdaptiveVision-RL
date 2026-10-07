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

- 当前基座为已合并 final SFT adapter 的
  `/root/autodl-tmp/models/qwen3vl_4b_sft_final_merged`。训练与评估共用模板适配：
  `add_generation_prompt=True` 只预填 `<|im_start|>assistant\n`，由模型在
  completion 中自行生成 `<think>`。
- 第一轮输入低分辨率全图和问题，策略在直接输出 `<answer>...</answer>` 与
  输出 `<tool_call>...</tool_call>` 请求高清局部图之间二选一。两种动作都必须
  以非空 `<think>...</think>` 开始；第二轮回答也必须如此。
- 裁剪工具接收 Qwen-VL 原生的 0～1000 归一化 `xyxy` 坐标；环境按低清图
  尺寸换算后映射到原图执行裁剪，参考框与 Coverage+IoU 仍使用 0～1 坐标。
- 合法裁剪请求会产生第二轮模型输出，第二轮不得再次调用工具。
- 第二轮同时输入低分辨率全图和高清局部图；vLLM 已显式设置为每个 prompt
  最多接收两张图片。
- SFT、rollout 和独立评测共用 `adaptive_vision_rl/images.py` 的预处理。
  每轮传入原始低清图和裁剪图，仅在构造模型输入时缩放一次；图像加载先应用
  EXIF 方向，确保低清图与高清原图的裁剪坐标一致。
- verl-agent 对每个 turn 单独前向，因此：
  - 唯一获取的视觉 token 数为 `low + crop`；
  - 两轮轨迹实际处理的视觉 token 数为 `low + (low + crop)`。
  两种口径都会记录。
- Outcome Reward 为答案分（数值接近可获部分分）、最高 0.1 的格式奖励和论文形式的 balance reward
  之和。只有包含非空且非占位符 `<think>...</think>` 的合法 action 才获得
  归一化格式分 1.0，否则为 0。直接回答的格式奖励为该分数乘 0.1，
  两轮工具轨迹先平均 tool call 与最终 answer 的格式分再乘 0.1。
  格式不合规的最终答案不获得答案分或准确率；两轮轨迹仍保留合法工具轮的格式分。
  日志中的 `answer_score` 保留原始相似度，非精确答案在构造 Outcome Reward 时
  乘以 `1 - balance_penalty`，避免近似错误答案因免付成本而超过精确答案。
  Balance penalty 从论文的 0.1 降为 0.01，阈值保持 0.2。
- PPO 使用论文的非对称裁剪范围：下界 0.20、上界 0.24；学习率为
  `1e-6`，不使用 KL，工具优势系数为 0.3。
- Tool Reward 对所有候选参考框取
  `sqrt(Coverage * IoU)` 的最大值。没有合格参考框的样本不计算工具优势。
- 一轮直接回答仅使用 `A_outcome`。两轮轨迹的完整第一轮输出使用
  `A_outcome + 0.3 * A_tool`，第二轮回答使用 `A_outcome`。
  这里的“完整输出”包括生成的 `<think>`、动作标签及其内容和结束 token；
  同一轮所有有效 response token 共用该轮优势，prompt 和 padding token 不参与 loss。
  第二轮生成的 `<think>` 与 `<answer>` 同样共用 `A_outcome`，不再加工具优势。
- PPO 分别对工具轮 token 和回答轮 token 做归一化。每个训练 step 生成
  64 条轨迹，展开后得到 64～128 个 turn row；不足 128 的部分使用零 loss
  padding 补齐。Padding 不参与奖励、优势、token 分母或训练指标。当前配置要求
  `ppo_mini_batch_size = 2 × data.train_batch_size × env.rollout.n`，以保证两个
  token 分母都在完整训练 step 上计算。
- 准确率保留项目已有的确定性精确匹配；结果奖励对接近的纯数字另给相对相似度部分分，不在训练期间调用在线
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
错误映射仍存在时修改当前进程的映射；重复调用会识别已修复的映射，不会叠加。
`scripts/run_dtpo_lora.sh` 设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，让
vLLM EngineCore 留在加载该 hook 的 Ray Actor 进程内，使 rank 64、alpha 128
的 LoRA 不会因错误映射挂载到视觉塔。启动日志中应出现：

```text
Applied the vLLM 0.11.2 Qwen3-VL LoRA mapping backport
```

这只覆盖已知的映射问题；A800 上仍须运行下文的两步 GPU Smoke Test，检查
vLLM 初始化、权重同步与完整训练 step。单独在父进程看到该日志不足以证明
另起的 EngineCore 子进程也应用了补丁。

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
loss 和 checkpoint 路径。单张 A800 80 GB 的 actor 更新与旧策略 log-prob 重算
均使用 micro batch 2，以保持两次前向的分组一致，避免无参数更新时出现额外
PPO ratio 偏差。128 个 row 的训练 step 和 DTPO loss 分母保持不变：

```bash
bash scripts/run_dtpo_lora.sh \
  trainer.total_training_steps=2 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  trainer.experiment_name=qwen3vl_4b_dtpo_smoke \
  trainer.default_local_dir=/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_smoke \
  trainer.rollout_data_dir=/root/autodl-tmp/outputs/rollouts/qwen3vl_4b_dtpo_smoke
```

默认配置直接读取已合并 final SFT adapter 的模型目录。启动前确认该目录包含
`config.json` 和完整模型权重；后续 DTPO adapter 评测默认读取同一目录。
如需切换其他 SFT 合并模型，可同时覆盖训练的
`actor_rollout_ref.model.path=...` 与评测的 `--model ...`。

工具调用后的第二轮会同时输入低清图和裁剪图。少数高分辨率样本可能使展开后的
视觉 token 超过 `data.max_prompt_length=8192`；collector 会仅对超限的观测
逐步缩小图片，并用处理后的同一组图片构造 vLLM 输入、actor 图像张量和视觉
token 统计。日志中的 `Resized overlong multimodal prompt` 会记录样本、轮次、
缩放前后 token 数和图片尺寸。`max_response_length=1024` 使总序列上限为 9216；
vLLM 的 `max_num_batched_tokens=10240` 覆盖这个长度。保留
`data.truncation=error`，避免截断视觉 token。
同步代码到 A800 后，可先运行
`python scripts/check_dtpo_image_budget.py --scan-data`，用实际 Qwen3-VL
processor 检查一张／两张大图、极细裁剪图，以及训练／验证数据中最长的两轮文本。
此检查无需启动 GPU 训练。只有图片超限时会缩图；若文本加两张最小图片仍超限，
collector 仍会报错，以免视觉 token 被截断。极细工具裁剪会补黑边至纵横比不超过
100，实际裁剪坐标和几何奖励保持原值。

Smoke test 需要重点确认：

- vLLM 能接收第二轮的两张图片；
- 64～128 个真实 turn row 能正确 padding 到 128；
- `loss_mask` 中 padding 权重为 0；
- LoRA 权重能在 FSDP Actor 和 vLLM rollout 间同步；
- 日志中出现 DTPO 与视觉 token 指标。

## 正式训练

```bash
bash scripts/run_dtpo_lora.sh
```

默认每 20 个训练 step 保存一次可恢复训练的 checkpoint，并只保留最近 1 个：

```yaml
trainer:
  save_freq: 20
  max_actor_ckpt_to_keep: 1
```

本项目在成功保存和恢复后还会清理旧 `global_step_*` 目录，以免 verl-agent 在
进程重启后遗留多余 checkpoint。checkpoint 含完整 FP32 actor 状态、
LoRA 的 AdamW 状态、数据加载器状态和配置；预计约 19～22 GB／个。
断点续训后首次保存前也会先删除旧 checkpoint，以免短时占用两份空间。
如果新 checkpoint 保存失败，这段时间内将无法从旧 checkpoint 恢复；需要从
SFT 合并模型重新开始训练。
首次保存后用 `du -sh /root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_lora/global_step_*`
核对实际大小。

每次保存 checkpoint（包括训练最后一步）也会自动保存当前 LoRA adapter：

```text
global_step_N/actor/lora_adapter/adapter_model.safetensors
global_step_N/actor/lora_adapter/adapter_config.json
```

adapter 使用 actor 实际的 LoRA 配置，默认 4B/rank 64 模型约有 504 个张量，
另占约 0.5 GB 磁盘空间，并随所属 checkpoint 一起轮换删除。评测时可直接将
训练输出根目录或指定 `global_step_N` 传给 `scripts/evaluate_dtpo.py --checkpoint`。

固定版 verl-agent 的分层收集器漏掉 Qwen3-VL 的 `language_model.layers`，
曾产生 17 B 空文件。项目改为在完整 FSDP 参数上下文内收集并复制 LoRA，
检查 A/B 配对、形状及是否漏收；权重和配置都写入成功后才发布 adapter 目录。
adapter 收集或写入失败只打印警告，训练继续；完整 `.pt` checkpoint 仍保留，
可后续离线提取 adapter。完整 checkpoint 本身保存失败仍会报错。
运行中的进程不会自动载入新代码，需重新启动训练进程才能应用。

旧 checkpoint 若只有 `.pt`，仍可按需离线导出，例如 run2 的 step 300：

```bash
python -m scripts.export_dtpo_lora \
  --checkpoint /root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_run2/global_step_300 \
  --output /root/autodl-tmp/models/qwen3vl_4b_dtpo_run2_step300_adapter \
  --expected-tensors 504
```

导出只读取模型 `.pt`，并用内存映射避免把 19 GB 模型文件全部复制到内存。
输出目录包含 `adapter_model.safetensors` 和 `adapter_config.json`；评测时将
此目录传给 `scripts/evaluate_dtpo.py --checkpoint`。若训练时通过命令行覆盖过
基础模型路径、LoRA rank 或 alpha，导出时也用 `--base-model`、`--rank`、`--alpha`
传入相同值；若用另一份配置文件启动，则用 `--config` 指向它。

验证在训练前和每 50 step 运行一次（最后一步也运行），使用固定 val 集、
`temperature=0` 的贪心生成和当前 actor 的内存中 LoRA；每次生成前由 FSDP
sharding manager 把 LoRA 参数同步到 vLLM；首次同步也会在基础权重之后立即
加载当前 LoRA，不读取 `lora_adapter` 导出文件。
因此空的独立 adapter 文件不会直接导致验证沿用旧权重。连续几次验证指标
完全相同，仍需看当前 run 的 `actor/grad_norm`、`actor/lr` 和验证样例输出，
才能区分参数没有更新与贪心输出尚未改变。可在重启时添加
`trainer.log_val_generations=16`，把部分验证样例写入 WandB 供逐次比较。

最后一个训练 step 无论能否被 20 整除都会保存。输出目录由
`trainer.default_local_dir` 控制。当前默认训练输出均写到 AutoDL 数据盘：

```text
/root/autodl-tmp/checkpoints/qwen3vl_4b_dtpo_lora  # 可恢复训练的 checkpoint
/root/autodl-tmp/outputs/rollouts/qwen3vl_4b_dtpo_lora  # rollout JSONL
/root/autodl-tmp/wandb  # WandB 本地日志
```

中断后先检查输出目录下的 `latest_checkpointed_iteration.txt`。存在时保留同一
`trainer.default_local_dir` 并使用 `trainer.resume_mode=auto`；不存在时只能从
第 0 步重启，rollout JSONL 和 WandB 日志不能恢复模型或优化器状态。

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
- `dtpo/answer_score`；
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
| `data.train_batch_size` | 8 | 每个训练 step 的不同问题数 |
| `env.rollout.n` | 8 | 每个问题在线采样的轨迹数 |
| 真实 trajectory 数 | 64 | `8 × 8` |
| 真实 turn row 数 | 64～128 | 直接回答 1 行，工具轨迹 2 行 |
| Padding 后 row 数 | 128 | Padding row 的 loss 为 0 |
| `ppo_mini_batch_size` | 128 | 覆盖完整的 step-expanded batch |
| `ppo_micro_batch_size_per_gpu` | 2 | 每次前向／反向处理的 row 数 |
| 梯度累积次数 | 64 | `128 ÷ 2` |
| `rollout.log_prob_micro_batch_size_per_gpu` | 2 | 旧策略 log-prob 和 entropy 的 micro batch，与 actor 更新一致 |

这个起点参考了本项目 SFT 的 micro batch 2。固定版本的 verl-agent 在 rollout
结束后调用 vLLM level-1 sleep，训练 actor 时不再同时保留 vLLM 权重和 KV cache；
但 DTPO 第二轮可能包含两张图片，因此尚不能仅凭 SFT 结果保证更大的 micro batch。
旧策略 log-prob 虽不反传，仍会计算逐 token entropy。step 300 的固定回答验证中，
旧概率按 4 条重算、更新按 2 条前向，在参数未更新时也出现了明显 ratio 偏差；
两者都按 2 条处理后，本次单图、双图和混合场景的有效 token log-prob 完全相同。
因此默认将两者统一为 2。此结果不代表 actor 与 vLLM 的概率差已经全部消失。

后续若调整 micro batch，应同时调整 actor 更新和旧概率重算，并重新验证同一
输入在无更新时的 ratio；单独调整其中一项可能重新引入偏差。若 OOM 发生在
vLLM 生成或模型初始化，micro batch 设置不会解决该阶段的峰值；应根据 OOM
堆栈和显存日志调整 vLLM 内存预算。

## 集成约束

`actor_rollout_ref.rollout.multi_turn.enable` 在 YAML 中必须保持为 `false`。
训练入口会在 verl-agent 完成配置校验和 Worker 初始化后，仅打开 Actor 使用
`loss_mask` 的分支；真正的两轮交互由 `AdaptiveVisionTrajectoryCollector` 和自定义
环境管理。

Entropy 和两条 KL 路径必须保持关闭，因为当前 `loss_mask` 存储的是 DTPO 精确
归一化权重，而不只是布尔 mask。当前实现限定单机单卡，不能直接通过增加 GPU 数量
扩展；多卡版本需要重新处理 mini-batch 归一化和跨 rank token 计数。

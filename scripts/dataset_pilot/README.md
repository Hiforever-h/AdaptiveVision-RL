# 20 条离线区域标注试验

本目录保存 Smart-Train → DeepSeek Flash 的先导脚本。本轮只处理 20 条，不生成正式 3000/300/500 划分。

正式数据来源固定为：

- RL 训练：Smart-Train 3,000 条。
- checkpoint / 调参：Smart-Train 独立留出约 300 条。
- 最终评测：Smart-Val 500 条，不能依据教师或策略答题表现重选。

## 运行

在项目根目录执行；环境名按用户指定保留为 **vison-rl**。

```bash
conda env create -f environment.yml  # 环境已存在时跳过
conda activate vison-rl
python -m scripts.dataset_pilot.prepare --count 20 --seed 20260921
python -m scripts.dataset_pilot.annotate --limit 1 --workers 1
python -m scripts.dataset_pilot.annotate --workers 3
python -m scripts.dataset_pilot.report
python -m unittest scripts.dataset_pilot.test_core -v
```

提示词配对实验使用 `variant.py` 冻结同一份manifest和图像到新目录，再通过 `--prompt` 指定提示词：

```bash
python -m scripts.dataset_pilot.variant \
  --source data/pilot20 \
  --output data/pilot20_v4 \
  --prompt scripts/dataset_pilot/prompt_v4.txt
python -m scripts.dataset_pilot.annotate \
  --output data/pilot20_v4 \
  --prompt scripts/dataset_pilot/prompt_v4.txt \
  --workers 3
python -m scripts.dataset_pilot.report \
  --output data/pilot20_v4 \
  --report-dir reports/pilot20_v4
```

像素坐标和 Z.AI GLM-5.3-Flash 运行示例：

```bash
python -m scripts.dataset_pilot.annotate \
  --output data/pilot20_v6_glm53_pixels \
  --prompt scripts/dataset_pilot/prompt_v5_pixels.txt \
  --coordinate-format pixels \
  --provider zai \
  --model glm-5.3-flash \
  --workers 3
```

GLM-5.3-Flash官方只支持启用thinking；`--thinking disabled`仅用于单条兼容性探测，当前服务端返回HTTP 400。降低开销时使用 `--thinking enabled --reasoning-effort low`，并为不同配置建立独立输出目录，避免混用缓存。

提示词结果见 `reports/PROMPT_COMPARISON.md`；DeepSeek与GLM教师对比见 `reports/MODEL_COMPARISON.md`。v4完整保留初版原文，只插入必要上下文段落；它没有覆盖默认的 `prompt.txt`，避免把尚未验证合格的版本设为正式默认。v3保留为发现严格对照差异前的中间实验。

本次环境已经创建，manifest 和 API 结果也已存在。`prepare` 遇到已有 manifest 会拒绝重新抽样；`annotate` 自动复用成功缓存。更新失效密钥后可显式使用 `--retry-errors` 重试认证等请求失败。
有成功 HTTP 响应但格式不合格时，保留原始返回并优先修复解析，不通过连续重新提问挑选正确答案。

DeepSeek 密钥从项目根目录 `.env` 的 `DEEPSEEK_API_KEY` 读取；Z.AI 密钥从 `ZAI_API_KEY` 读取，均不进入 Git。
默认模型分别为 `deepseek-flash` 和 `glm-5.3-flash`，可用 `DEEPSEEK_MODEL`／`ZAI_MODEL` 或 `--model` 设置。
脚本只把对应密钥发送到所选服务的官方端点，关闭重定向；缓存不保存请求头、密钥或图像 Base64。
成功请求含原图与问题，不含标准答案。接口可能对图像进行内部缩放；本地原图尺寸保持不变。

## 文件

| 文件 | 用途 |
|---|---|
| `prepare.py` | 从 8 个 HF viewer 窗口取候选，固定抽样并下载原图 |
| `prompt.txt` | 固定的英文区域标注提示词 |
| `annotate.py` | 调用 API、缓存、结构校验、答案自动核对 |
| `common.py` | 坐标约定、像素转换、答案比较、文件读写 |
| `reward.py` | 纯本地 coverage / IoU 奖励，无 LLM 调用 |
| `report.py` | 原图框选、真实裁剪、HTML 报告与 JSONL 导出 |
| `test_core.py` | 奖励、无效坐标、答案误匹配、缓存和答案泄漏检查 |

输出：

```text
data/pilot20/                         # 已加入 .gitignore
  manifest.jsonl                     # 固定20条；pilot_train_reserved
  sampling.json                      # 来源版本观测、窗口、seed和局限
  source_cache/                      # HF原始行数据
  originals/                         # 原图原始字节，未修改
  images/                            # 保持尺寸，统一EXIF方向与RGB的PNG
  lowres/                            # 按原数据tgt尺寸缩小的全图，仅预览
  api_cache/                         # 原始模型响应、调用用量、耗时
  annotations.jsonl                  # 自动解析结果及严格答案比较
  reviewed_annotations.jsonl         # 合并Codex目视复核，不改变原框
  accepted_by_answer.jsonl           # 仅依据答案正确导出；不保证定位正确
  geometry_diagnostics.jsonl         # 参考裁剪与整图裁剪的奖励诊断
  previews/                          # 原图框选、实际裁剪、20条联系表
reports/pilot20/
  index.html                         # 可直接打开的自包含交互报告
  RESULTS.md                         # 自动生成的统计和逐条答案
  summary.json                       # 机器可读统计
  visual_review.jsonl                # Codex看图复核；不是人工真值
  FINDINGS.md                        # 实验观察与局限
```

## 标注与接受规则

模型仅输出 `status`、`reference_boxes`、`answer_from_image`；首轮返回一个相对于原图左上角的 xyxy 局部框。
默认使用归一化坐标；`--coordinate-format pixels` 要求模型返回原图整数像素坐标，并确定性转换为归一化框供奖励使用。
归一化框转像素时左上角 floor、右下角 ceil；像素模式直接使用模型返回的整数边界。右／下边界均按 exclusive 处理，与 Pillow crop 一致。
返回全局或不确定状态时框可以为空。全图框不强行丢弃。
兼容模型偶尔回显的 `"type": "json_object"`，仅删除这个已知冗余字段，并在质量记录中注明；不修正坐标或答案。

不包含 `evidence_description`，不做 crop-only 盲答验证。是否要求必要上下文由每次运行冻结保存的提示词决定；v4/v5明确要求表格行列标题和图表标签、图例或坐标轴。
自动比较只使用大小写／空格／Unicode规范化，以及数值精确比较；不粗暴删除单位、百分号或使用子串匹配。
人工或 Codex 复核可补充单位与共同作者等语义差异，本次复核者明确记录为 Codex，而非人工。
`annotations.jsonl` 保留自动比较，`reviewed_annotations.jsonl` 的 `accepted_by_answer` 按复核更新，自动比较仍在 `answer_match` 中保留。
定位检查只提供诊断，不在本轮悄悄改变用户指定的答案正确筛选口径。

## 几何奖励

```text
coverage = intersection(predicted, reference) / area(reference)
IoU = intersection / union
reward = coverage ** 0.5 * IoU ** 0.5
```

参考框完全匹配奖励为 1；无交集为 0；参考区域占全图 4% 时，裁整图得 0.2。
没有参考框返回 `None`，由训练侧 mask；无效坐标报错。输入应为工具实际执行的框，不能以尚未裁剪／修正的请求坐标判分。
面积加权、调用成本、直接回答路径和 DTPO 优势计算尚未接入训练。本函数不包含这些策略。
**几何公式正确不代表伪标签正确**，首轮发现的漏框会直接误导该奖励，详见实验报告。

## 抽样范围与后续全量构造

数据浏览器当前只公开约 18,761 条转换结果，`partial=true`。本轮从其中 8 个窗口、800 条候选中，
在 `use_tool=false/true` 两组各取 10 条；这不是全量随机分层样本，工具标记也不等于当前策略的必须动作。
所选 `use_tool=true` 样本集中在一个图表窗口，不能把本轮正确率外推为全量效果。
仅保存 viewer 提供的原始图像分辨率，不声称恢复了上游任务未提供的更高清图片；其中存在 88×34 的文字图。

HF viewer 不能按源 commit 固定，本轮记录下载前后观测到的 commit，并缓存每一行、原始图片和哈希以复现。
已经在20条内部按doc_id、像素SHA256和dHash近重复检查；没有冒称完成全量防泄漏。
正式扩大前需要完整读取Smart-Train与Smart-Val，确认doc_id的实际含义，结合来源、文档、精确／近重复图像做分组审计。
保留先导样本的训练候选归属，然后在训练池分组留出开发集。最终评测只从Smart-Val抽取500条并冻结；
遇到与训练集重合的组应优先从训练候选移除，而非按最终评测表现替换测试题。

参考：[Smart-Train](https://huggingface.co/datasets/Senqiao/VisionThink-Smart-Train)、[Smart-Val](https://huggingface.co/datasets/Senqiao/VisionThink-Smart-Val)、[DeepSeek Vision](https://api-docs.deepseek.com/guides/vision/)。

# AdaptiveVision-RL

基于 Qwen3-VL-4B-Instruct 与 GRPO 的单卡自适应视觉获取项目。模型先读取低分辨率全图，并在细节不足时调用局部裁剪工具补充高分辨率视觉信息，在保持视觉问答能力的同时降低视觉 token 消耗。总体范围见 [PLAN.md](PLAN.md)。

项目已完成数据构建、区域标注、GRPO 训练和基准评测。正式数据冻结为 3,000 条 Train、300 条 Val 和 500 条 Test；Train 按原数据 `use_tool` 提示分层，低清即可回答与建议使用高清各 1,500 条。教师模型使用 `glm-5.3-flash`、`reasoning_effort=max` 和像素坐标，在线工具奖励采用 Coverage、IoU 与裁剪面积约束的组合。

- [脚本及复现方式](scripts/dataset_pilot/README.md)
- [区域标注试验结果](reports/pilot20/RESULTS.md)
- [实验观察](reports/pilot20/FINDINGS.md)
- [三版提示词配对对照](reports/PROMPT_COMPARISON.md)
- [可视化报告](reports/pilot20/index.html)

训练采用 LoRA 与 GRPO，仅更新低秩适配参数而不进行全参数训练；同时接入两轮裁剪环境、Coverage + IoU 工具奖励、0.01 工具调用成本和 Qwen3-VL 视觉 token 统计。推理阶段最多进行一次局部裁剪；若低分辨率图像已经包含足够信息，模型直接作答。

## 实验结果

在相同的 Qwen3-VL-4B-Instruct 初始化、训练数据、GRPO 超参数和解码设置下，对比以下两种视觉输入策略：

- **Base 高清图 + GRPO**：始终输入完整高清图，作为 100% 视觉 token 基线。
- **低分辨率 + 局部裁剪 + GRPO**：先输入 1/4 分辨率全图，模型仅在需要细节时从原始高清图中裁剪一个局部区域。

| 方法 | ChartQA（test） | OCRBench（test） | DocVQA（val） | MME（test） | MMVet（test） | RealWorldQA（test） | POPE（test） | MathVista（testmini） | MathVerse（testmini） | 视觉 token ↓ | 相对平均性能 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base 高清图 + GRPO | 81.3 | 80.1 | 92.6 | 2270 | 60.6 | 68.7 | 85.6 | 72.8 | 61.9 | 100% | 100.0% |
| **低分辨率 + 局部裁剪 + GRPO** | **74.8** | **74.2** | **90.4** | **2294** | **60.0** | **65.4** | **85.6** | **68.4** | **58.8** | **57%** | **96.3%** |

其中 OCRBench 按满分 100 归一化，MME 报告总分；“相对平均性能”先分别计算各基准相对 Base 的得分比例，再取宏平均，避免 MME 的量纲主导平均值。

低分辨率全图配合一次自适应局部裁剪将平均视觉 token 消耗从 100% 降至 57%，减少 **43%**；九项基准的宏平均相对性能保持在 **96.3%**。细粒度文字、图表和视觉数学任务对分辨率更敏感，因此 ChartQA、OCRBench 与 MathVista 有一定下降；DocVQA、MMVet 和 POPE 基本保持，高效方案在 MME 上略有提升。整体上，该策略以约 3.7% 的相对平均性能差换取 43% 的视觉 token 节省。

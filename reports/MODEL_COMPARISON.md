# 区域标注教师模型配对实验

日期：2026-09-21。本对照使用相同的20条冻结样本、原图、问题、像素坐标提示词和 temperature=0。两轮提示词 SHA256 均为 `5acb56398705d3350427fdf84211cd63403566afe71024ba61e27bfffea5f580`，manifest 逐字节一致，标准答案和另一模型输出均未发送给教师。

DeepSeek 使用 `deepseek-flash`、关闭thinking；Z.AI 使用 `glm-5.3-flash`。GLM-5.3-Flash 的[官方模型说明](https://docs.z.ai/guides/vlm/glm-5.3-flash)不允许关闭thinking，因此本轮按官方支持范围启用thinking并设 `reasoning_effort=max`。这不是只改变权重的严格单变量实验，但数据、提示词和输出坐标格式保持一致。请求通过[官方Chat Completion端点](https://docs.z.ai/api-reference/llm/chat-completion)发送。

## 无thinking探测

随后对同一首条样本显式发送 `thinking.type=disabled`，并省略 `reasoning_effort`。Z.AI在0.535秒后返回HTTP 400，没有生成回答、坐标或计费token记录；因此没有继续调用其余19条。当前GLM-5.3-Flash不能用无thinking模式运行，后续若降低推理开销只能在官方支持的 `low`、`high`、`max` 中选择。

## 结果

| 指标 | DeepSeek Flash | GLM-5.3-Flash |
|---|---:|---:|
| 有效JSON和像素框 | 20/20 | 20/20 |
| 严格答案匹配 | 14/20 | 6/20 |
| Codex看图复核后答案正确 | 19/20 | 20/20 |
| 目标与必要上下文均可见 | 5/20 | 15/20 |
| 目标可见但上下文不足 | 3/20 | 2/20 |
| 目标文字被截断 | 4/20 | 1/20 |
| 目标细节完全落在框外 | 6/20 | 1/20 |
| 整图框 | 1/20 | 1/20 |
| 答案错误，定位不适用 | 1/20 | 0/20 |
| 框面积中位数 | 4.10% | 13.08% |
| API报告总token | 15,941 | 27,739 |
| API报告completion token | 703 | 7,827 |
| 单请求平均耗时 | 0.89秒 | 9.69秒 |

GLM在这20条上把“目标与必要上下文均可见”从5条提高到15条，并把目标完全漏框从6条降到1条。改善最明显的是表格行列定位、尺子与被测物共同覆盖，以及图表数值和年份共同覆盖。

GLM的框明显更大，配对框的IoU中位数仅0.259，说明它不只是把DeepSeek框轻微外扩，而是重新选择了区域。较大的框提高了上下文覆盖，也会降低几何奖励对精细裁剪的区分度；后续应在独立样本上验证面积与任务收益的权衡。

严格答案匹配从14条降到6条不是答题能力下降。GLM经常输出单位、解释或括号，例如把`Friday`写成`Friday, with 45,547 weddings...`；目视语义复核20条均正确。正式扩大前需要约束或解析这一输出格式，否则保守的自动匹配会误拒大量可用框。

代价也更高：本轮GLM平均延迟约为DeepSeek的10.9倍，总token约为1.74倍；6,896个token属于GLM报告的reasoning token。该成本发生在离线伪标签构造阶段，不会进入RL训练循环，但会影响3,300条正式标注的时间和预算。

## 仍失败的GLM样本

- `smart-train-00030`：完整框住Friday，但没有Saturday，不能仅凭裁剪确认“第二名”。
- `smart-train-05552`：包含2018/19数值和年份，但缺Boys/Girls图例。
- `smart-train-05519`：包含目标行数值，左边界切掉年龄标签中的`25`。
- `smart-train-05561`：包含November 2020及目标点，上边界裁掉`103.7`数值。
- `smart-train-09505`：原图只有88×34，整图就是合理区域，没有局部裁剪空间。

## 判断

就当前20条的区域伪标签质量，GLM-5.3-Flash显著优于DeepSeek Flash，值得作为下一轮候选教师。当前证据仍不足以直接生成全部3,300条：样本来自HF部分转换数据，复核由Codex完成，而且GLM配置因模型约束包含thinking。建议先把答案输出收紧，再从未参与本轮的Smart-Train样本抽取50至100条做盲复核；若完整目标与上下文比例仍接近本轮，再扩大构造。

## 产物

- GLM报告：`reports/pilot20_v6_glm53_pixels/index.html`
- GLM逐条复核：`reports/pilot20_v6_glm53_pixels/visual_review.jsonl`
- GLM原始响应：`data/pilot20_v6_glm53_pixels/api_cache/`
- DeepSeek像素版报告：`reports/pilot20_v5_pixels/index.html`
- 共用提示词：`scripts/dataset_pilot/prompt_v5_pixels.txt`

目视复核均由Codex完成，不是独立人工标注或人工真值。

# 20 条区域标注先导试验 · pilot20_v6_glm53_pixels

- 样本数：20；有效标注：20；错误：0；待标注：0。
- 答案自动匹配：6；按答案正确口径接受且有框：20。
- Codex 目视复核：20 条；复核后答案正确：20 条。复核不是独立模型评测或人工真值。
- 定位诊断：目标与必要上下文均可见 15 条、上下文不足 2 条、漏掉目标细节 1 条、目标文字被截断 1 条、整图框 1 条、答案错误不适用 0 条。诊断不改变按答案正确筛选的口径。
- 服务／模型：zai / glm-5.3-flash。
- API 报告用量：`{"prompt_tokens": 19912, "completion_tokens": 7827, "total_tokens": 27739, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}`。不把认证失败算成成功标注。
- 接受依据：严格匹配或独立记录的 Codex 目视复核；原始答案、框和 API 返回全部保留，不通过重试挑选正确答案。
- 不发送标准答案给教师，不做裁剪图独立回答验证；必要上下文要求以本次保存的 `prompt.txt` 为准。
- 框是伪标签；回答正确并不证明定位准确。本报告不宣称人工通过率；逐条复核见 `visual_review.jsonl`。

## 数据范围

本轮是 HF 数据浏览器部分转换数据的诊断性样本。8 个窗口共 800 条候选，固定 seed=20260921，
原 `use_tool` 标记各取 10 条；在抽样前按窗口轮转，并在样本内检查文档 ID、相同图片与 dHash 近重复。
原始字节、处理后图片、来源行号、采样清单及哈希均保存在 `/Users/hiforever/Documents/AiProject/AdaptiveVision-RL/data/pilot20_v6_glm53_pixels`。
尚未完成全量原始数据扫描、文档标识语义确认或 Train/Val 跨集合去重，也未构造正式 3000/300/500 划分。
正式方案：Smart-Train → 3000 训练 + 300 开发；Smart-Val → 500 最终评测。

## 逐条结果

| 样本 | 状态 | 标准答案 | 教师回答 | 框面积占比 |
|---|---|---|---|---|
| smart-train-00030 | 答案正确（Codex 复核） | Friday | Friday, with 45,547 weddings (after Saturday&#x27;s 122,173) | 4.6% |
| smart-train-08592 | 答案正确（Codex 复核） | 4.73 | 4.73 billion U.S. dollars | 34.5% |
| smart-train-09564 | 答案正确（Codex 复核） | A | (a) The image is corrupt, the specific corruption type is Defocus blur | 32.9% |
| smart-train-11026 | 答案正确（Codex 复核） | Leigh Bardugo | Leigh Bardugo | 7.6% |
| smart-train-13037 | 答案正确（Codex 复核） | 11.6 cu.ft. | 11.6 Cu.Ft. | 13.5% |
| smart-train-16069 | 答案正确（Codex 复核） | 2 | 2 | 52.5% |
| smart-train-18471 | 答案正确（Codex 复核） | 13.3 | 13.3 | 63.8% |
| smart-train-00004 | 答案正确（Codex 复核） | 2720.3 | 2,720.3 million U.S. dollars | 9.3% |
| smart-train-08568 | 答案正确（Codex 复核） | Camil Muscalu | Camil Muscalu and Wilhelm Schlag | 10.5% |
| smart-train-09505 | 答案正确（Codex 复核） | 82.50 | 82.50 | 100.0% |
| smart-train-05552 | 答案正确（Codex 复核） | 105357 | 105,357 boys participated in high school sports in 2018/19. | 14.5% |
| smart-train-05592 | 答案正确（Codex 复核） | 42138 | 42,138 million DKK | 12.7% |
| smart-train-05555 | 答案正确（Codex 复核） | 14475 | 14,475 hospital beds | 9.9% |
| smart-train-05547 | 答案正确（Codex 复核） | 68 | 68% | 19.8% |
| smart-train-05519 | 答案正确（Codex 复核） | 25 to 29 | 25 to 29 years (about 23.5 million total residents) | 3.0% |
| smart-train-05524 | 答案正确（Codex 复核） | 79.5 | 79.5 billion Indian rupees | 35.1% |
| smart-train-05580 | 答案正确（Codex 复核） | 10914 | 10,914 million U.S. dollars (in 2019) | 8.1% |
| smart-train-05561 | 答案正确（Codex 复核） | 103.7 | 103.7 | 10.5% |
| smart-train-05593 | 答案正确（Codex 复核） | 6.1 | 6.1% | 8.5% |
| smart-train-05534 | 答案正确（Codex 复核） | 278.04 | Iran&#x27;s national debt in 2019 was about 278.04 billion U.S. dollars. | 19.4% |

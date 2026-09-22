# AdaptiveVision-RL

基于 AdaptVision / DTPO 的单卡适配项目。总体范围见 [PLAN.md](PLAN.md)。

当前已完成 20 条 Smart-Train 区域标注试验，并将正式未标注数据冻结为
3,000 条 Train、300 条 Val 和 500 条 Test。Train 按原数据 `use_tool` 提示分层，
低清即可回答与建议使用高清各 1,500 条。后续教师默认使用 `glm-5.3-flash`、
`reasoning_effort=max` 和像素坐标；在线工具奖励计划采用 coverage 与 IoU 的几何组合。

- [脚本及复现方式](scripts/dataset_pilot/README.md)
- [实验结果](reports/pilot20/RESULTS.md)
- [实验观察](reports/pilot20/FINDINGS.md)
- [三版提示词配对对照](reports/PROMPT_COMPARISON.md)
- [可视化报告](reports/pilot20/index.html)

环境名为 `vison-rl`。项目 `.env` 存放 DeepSeek 密钥，已排除出 Git。
均衡版正式数据位于 `data/visionthink_3000_300_500_balanced/`。Val 300 与 Test 500 已使用
`glm-5.3-flash`、`reasoning_effort=max` 完成像素框标注；Train 3,000 尚未标注，RL 训练尚未开始。

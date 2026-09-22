# AdaptiveVision-RL

基于 AdaptVision / DTPO 的单卡适配项目。总体范围见 [PLAN.md](PLAN.md)。

当前已完成 20 条 Smart-Train 区域标注试验，并将正式数据冻结为
3,000 条 Train、300 条 Val 和 500 条 Test。Train 按原数据 `use_tool` 提示分层，
低清即可回答与建议使用高清各 1,500 条。后续教师默认使用 `glm-5.3-flash`、
`reasoning_effort=max` 和像素坐标；在线工具奖励计划采用 coverage 与 IoU 的几何组合。

- [脚本及复现方式](scripts/dataset_pilot/README.md)
- [实验结果](reports/pilot20/RESULTS.md)
- [实验观察](reports/pilot20/FINDINGS.md)
- [三版提示词配对对照](reports/PROMPT_COMPARISON.md)
- [可视化报告](reports/pilot20/index.html)

环境名为 `vison-rl`。项目 `.env` 存放 DeepSeek 密钥，已排除出 Git。
均衡版正式数据位于 `data/visionthink_3000_300_500_balanced/`。Train 3,000、Val 300 与 Test 500 已使用
`glm-5.3-flash`、`reasoning_effort=max` 执行像素框标注。Train 原有 46 条无效记录已按原分层和去重规则替换，
当前有效标注为 3,000/3,000；替换记录见 `reports/train_replacement_20260923/RESULTS.md`，全量可编辑框复核页面见
`reports/train3000_glm53_max_full_review/index.html`。首批 22 条手动框修改已合并，记录见
`reports/train_manual_box_edits_20260922/RESULTS.md`。RL 训练尚未开始。

DTPO + LoRA 训练代码已接入 verl-agent，包括两轮裁剪环境、Coverage+IoU
工具奖励、0.01 balance cost、turn-level advantage 与独立 loss 归一化，以及
Qwen3-VL 视觉 token 统计。安装、数据转换和启动方式见
[docs/DTPO_VERL_AGENT.md](docs/DTPO_VERL_AGENT.md)。默认配置位于
`configs/dtpo_qwen3vl_4b_lora.yaml`；提交代码不自动启动训练。

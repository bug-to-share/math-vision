# 多模态数学推理后训练

[HTML 简历展示](index.html) · [复现指南](docs/reproduce.md) · [评测结果与消融](docs/experiments.md)

几何模型可能读出 `30°`，却把它绑定到错误的角。只给最终答案打分，很难区分读图错误、推导错误和回答截断。这个项目从 Geometry3K 的对照诊断出发，在 Qwen3-VL-4B 上研究两种训练信号。

| 训练信号 | veRL 中的实现 | 状态 |
|---|---|---|
| 图像遮挡概率对照 | 原图生成回答，固定回答前缀比较原图与随机遮图的 token 概率。旧策略遮图 log-prob 在首次更新前缓存 | 双卡全参数 120 步已完成，四集同口径评测 |
| 感知与推理解耦 | 分开验证证据充分性与最终答案，按验证结果调整失败轨迹的感知 token 优势 | 标准 GRPO 实现与 CPU 验证完成，GPU 训练成绩待补 |

与 DAPO 的第 120 步 checkpoint 相比，加入遮挡对照与配套 NLL 后，**GeoQA 69.46% → 74.23%（+4.77 个百分点）**，**Geo3K 71.99% → 76.21%（+4.22 个百分点）**。四集共 2,645 题，成绩为三个共同解码种子的平均准确率。MathVision 的差异为 −0.22 个百分点。简单事实探针尚未检出读图改进。 [完整结果与限制](docs/experiments.md)

`overlay/` 包含两个训练 recipe、配套 DAPO 基线和评测代码。`patches/` 提供固定 veRL 版本所需的兼容改动。`index.html` 是独立展示页，下载后可直接用浏览器打开。

双方法的交互用四组匹配训练估计。四组分别启用两项中的零项、单项或两项，并对困难事实正确率与最终解题准确率分别估计差值。 [消融设计](docs/experiments.md#消融设计)

## 方法来源

- 原图与遮挡图的概率对照参考 [PAPO](https://arxiv.org/abs/2507.06448) 和[作者实现](https://github.com/MikeWangWZHL/PAPO)。本项目记录了 Qwen3-VL 的遮挡与缓存实现以及自己的评测协议。
- 独立证据验证、SVV 与优势路由参考 [MoCA](https://arxiv.org/abs/2605.14054)。本地版本采用均值与标准差归一化的 GRPO。证据提示词改编自论文，按原文 CC BY 4.0 署名。
- 代码基于 [veRL](https://github.com/verl-project/verl) 和 [verl-recipe](https://github.com/verl-project/verl-recipe)，精确版本见 [UPSTREAM.json](UPSTREAM.json)。保留上游代码版权及 [Apache-2.0 许可](LICENSE)。

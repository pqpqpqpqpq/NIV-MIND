# 论文分析覆盖说明

本目录的模型、训练和分析代码覆盖论文正文图 2–5 以及补充表 S1–S9 所涉及的
计算环节。

| 论文内容 | 对应代码 |
|---|---|
| NIV-MIND 网络、五折训练和最终推理 | `model.py`、`train.py`、`predict.py` |
| 五折 OOF、AUC、AUPRC、ROC、PR、DCA | `analysis/comparison.py`、`analysis/metrics.py` |
| NIV-MIND-C、NIV-MIND-F、LightGBM、HACOR、ROX | `analysis/comparison.py` |
| 起始期使用终点期模型评估 | `analysis/phase_evaluation.py` |
| 疾病或临床亚组 AUC 与置信区间 | `analysis/subgroup.py` |
| 起始对齐和终点对齐的时间地标选择 | `analysis/landmarks.py` |
| 5、10、20 分钟观察窗口 | `NIV_MIND_ablation_models` 的 `window` 组 |
| ST-GCN、AltFormer、ST、TS 结构对照 | `NIV_MIND_ablation_models` 的 `architecture` 组 |
| 1、2、5、10、30 秒采样 | `NIV_MIND_ablation_models` 的 `sampling` 组 |
| 单变量序列替换为样本内均值 | `analysis/temporal_ablation.py` |
| 41 项特征、328 项全序列候选、2,048 项窗口候选 | `analysis/engineered_features.py` |
| 变量—窗口贡献分数和 Top-5 窗口 | `analysis/temporal_attribution.py` |
| 512 维细粒度表征提取 | `analysis/representations.py` |
| 2,000 次配对 bootstrap ΔAUC | `analysis/metrics.py` |
| 缺失率、插值、训练折标准化 | `analysis/preprocessing.py` |

## 数据边界

主交付数据为每 2 秒一个点的 600 点序列。论文的 1 秒实验需要每例 1200 个真实
原始点，注册为 `sampling/sample_1200`。该变体可训练，但不能从 600 点序列插值
得到，也未附带五折权重。

连续时间地标分析需要每个窗口的 NIV 起始时间、观察终点和窗口结束时间。
`analysis/landmarks.py` 提供论文中的 0.5 小时容差选择规则，但这些字段不在最终
模型输入 PTH 中，运行时须提供窗口级 CSV。

疾病亚组分析需要患者或 admission 到疾病类别的映射。为避免把受保护的临床表格
放入代码包，`analysis/subgroup.py` 通过外部 CSV 接收该映射。

## 结构说明

论文方法文字将粗粒度、门控和融合向量概括为 512 维简单门控；已发布 PTH 的实际
可执行结构使用 128 维粗粒度编码、256 维融合表示和双向交叉注意力门控。
交付代码以 PTH 的实际结构为准，因为它能够严格加载权重并复现已保存模型输出。

当前交付的完整 NIV-MIND 权重可重算论文 endpoint 结果（OOF AUC 约 0.952、
AUPRC 约 0.897）。论文图 3 所用 NIV-MIND-C 和 NIV-MIND-F 的特定五折权重未在
本地归档中完整保留；目录中已提供两者的模型定义和五折训练入口。若已有论文归档的
长表或宽表 OOF 预测，可通过 `analysis/comparison.py --prediction-table <csv>`
载入，重新生成相同的指标和曲线。

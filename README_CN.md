# NIV-MIND

[English](README.md)

NIV-MIND 使用连续呼吸机测量值和临床变量预测无创通气结局。本仓库包含模型结构、
五折训练流程、推理工具及论文图表对应的分析代码。

## 目录结构

```text
model_new/
├── NIV_MIND_model/
│   ├── model.py                 # NIV-MIND 主模型
│   ├── backbone.py              # ST-GCN 与 AltFormer
│   ├── fusion.py                # 临床编码器与多模态融合
│   ├── dataset.py               # 数据读取与校验
│   ├── train.py                 # 五折训练
│   ├── predict.py               # 推理入口
│   ├── analysis/                # 论文指标与统计分析
│   └── tests/                   # 主模型测试
└── NIV_MIND_ablation_models/
    ├── compact.py               # 论文中的结构与时间分辨率变体
    ├── registry.py              # 实验注册表
    ├── dataset.py               # 实验数据接口
    ├── train.py                 # 统一训练入口
    ├── predict.py               # 统一推理入口
    └── tests/                   # 实验模型测试
```

数据文件及字段约定见 `data_schema.json`。

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
```

Linux 或 macOS：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 数据格式

仓库不包含研究数据。运行时按以下结构放置文件：

```text
data/
├── data_failure.pkl
├── data_success.pkl
├── data_fine_mean.npy
├── data_fine_std.npy
├── data_coarse_mean.npy
└── data_coarse_std.npy
```

`data_failure.pkl` 和 `data_success.pkl` 中的每条记录均为字典，字段如下：

- `fine`：形状为 `[600, 8]` 的数值数组；也接受含第九列时间戳的数组，读取时会
  删除时间戳列；
- `coarse`：形状为 `[20]` 的临床特征向量。

程序按照输入文件分配标签：成功为类别 `0`，失败为类别 `1`。

## 主模型训练

运行全部五折：

```bash
python -m model_new.NIV_MIND_model.train
```

只运行一个 fold：

```bash
python -m model_new.NIV_MIND_model.train --fold 0
```

默认配置为 Adam 优化器、初始学习率 `1e-4`、余弦退火、逆频率类别权重、最多
200 个 epoch，并根据验证集 AUC 进行早停，patience 为 25。

每折最佳模型保存为纯张量状态字典，训练指标单独保存为 JSON。

## 模型推理

模型权重单独提供，不保存在本仓库。将权重放入本地目录后运行：

```bash
python -m model_new.NIV_MIND_model.predict \
  --checkpoint weights/main/fold_0_best.pth \
  --input data_input.npz \
  --output predictions.csv
```

NPZ 输入必须包含：

- `fine`：`float32 [N, 600, 8]`；
- `coarse`：`float32 [N, 20]`。

## 论文展示的实验

仓库仅保留论文正文和补充材料中展示的实验。

### 观察窗口长度

| 变体 | 输入点数 | 2 秒采样下的观察时长 |
|---|---:|---:|
| `window_150` | 150 | 5 分钟 |
| `window_300` | 300 | 10 分钟 |
| `full` | 600 | 20 分钟 |

### 模型结构

| 变体 | 含义 |
|---|---|
| `no_altformer` | 仅保留 ST-GCN 表征 |
| `no_stgcn` | 不使用图卷积的 AltFormer 表征 |
| `only_st` | 仅保留空间到时间分支 |
| `only_ts` | 仅保留时间到空间分支 |
| `full` | 完整双分支结构 |

### 采样间隔

| 变体 | 输入点数 | 采样间隔 |
|---|---:|---:|
| `sample_1200` | 1200 | 1 秒 |
| `full` | 600 | 2 秒 |
| `sample_240` | 240 | 5 秒 |
| `sample_120` | 120 | 10 秒 |
| `sample_40` | 40 | 30 秒 |

1 秒变体必须使用原生 1,200 点序列，程序不会从 600 点输入构造 1 秒观测值。

### 单变量时间分析

`analysis/temporal_ablation.py` 会依次将八个呼吸机变量中的一个替换为该样本对应
变量的时间均值，其余七个变量保持不变。

### 运行论文实验

例如运行空间到时间单分支实验：

```bash
python -m model_new.NIV_MIND_ablation_models.train \
  --group architecture \
  --variant only_st
```

有效实验组为：

- `architecture`
- `sampling`
- `window`
- `compact_baseline`

## 论文指标与图表数据

生成五折 OOF 指标、ROC 曲线、PR 曲线和决策曲线：

```bash
python -m model_new.NIV_MIND_model.analysis.comparison \
  --main-weights weights/main \
  --comparator-weights weights/comparators \
  --output-dir paper_results/endpoint_comparison
```

分析模块还包括：

- AUC、AUPRC、敏感度、特异度及 Youden 阈值；
- 非参数 bootstrap 置信区间及配对 AUC 差值；
- NIV-MIND-C、NIV-MIND-F、LightGBM、HACOR 和 ROX 对照；
- 临床亚组分析；
- 起始对齐及终点对齐时间地标；
- 变量—时间窗口贡献分数；
- 41 项信号描述符、328 项全序列候选和 2,048 项窗口候选；
- 细粒度表征提取。

使用匹配的五折权重和 1,139 条评估记录可得到：

| 指标 | pooled OOF 结果 |
|---|---:|
| AUC | 0.952382 |
| AUPRC | 0.897391 |

亚组分析和连续时间地标分析还需要单独提供对应的受保护元数据。

## 测试

```bash
pytest model_new/NIV_MIND_model/tests \
       model_new/NIV_MIND_ablation_models/tests -q
```

若本地没有单独提供的模型权重，依赖权重的测试会自动跳过。

## 不纳入版本管理的文件

仓库排除以下内容：

- 模型权重：`.pth`、`.pt`、`.ckpt`；
- 研究数据：`.pkl`、`.pickle`、`.npy`、`.npz`、`.csv`、Parquet 和 HDF5；
- `data/` 目录；
- 预测结果、分析输出及缓存文件。

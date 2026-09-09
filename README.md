# NIV-MIND

[中文说明](README_CN.md)

NIV-MIND predicts non-invasive ventilation outcomes by combining continuous
ventilator measurements with clinical variables. This repository contains the
model architecture, five-fold training pipeline, inference utilities, and the
analyses reported in the manuscript.

## Repository structure

```text
model_new/
├── NIV_MIND_model/
│   ├── model.py
│   ├── backbone.py
│   ├── fusion.py
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── analysis/
│   └── tests/
└── NIV_MIND_ablation_models/
    ├── compact.py
    ├── registry.py
    ├── dataset.py
    ├── train.py
    ├── predict.py
    └── tests/
```

`data_schema.json` describes the input files and sample fields.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Data are not included in the repository. Place the files in the following
layout:

```text
data/
├── data_failure.pkl
├── data_success.pkl
├── data_fine_mean.npy
├── data_fine_std.npy
├── data_coarse_mean.npy
└── data_coarse_std.npy
```

Each record in `data_failure.pkl` and `data_success.pkl` is a dictionary with:

- `fine`: a numeric array with shape `[600, 8]`; a ninth timestamp column is
  also accepted and removed by the loader;
- `coarse`: a numeric vector with shape `[20]`.

Labels are assigned from the input file: success is class `0` and failure is
class `1`.

## Main model training

Run all five folds:

```bash
python -m model_new.NIV_MIND_model.train
```

Run one fold:

```bash
python -m model_new.NIV_MIND_model.train --fold 0
```

The default configuration uses Adam, an initial learning rate of `1e-4`,
cosine annealing, inverse-frequency class weights, up to 200 epochs, and early
stopping on validation AUC with a patience of 25 epochs.

The best model from each fold is written as a tensor-only state dictionary.
Metrics are stored separately as JSON.

## Inference

Model weights are distributed separately and are not committed to this
repository. After placing a fold checkpoint in a local directory, run:

```bash
python -m model_new.NIV_MIND_model.predict \
  --checkpoint weights/main/fold_0_best.pth \
  --input data_input.npz \
  --output predictions.csv
```

The NPZ input must contain:

- `fine`: `float32 [N, 600, 8]`;
- `coarse`: `float32 [N, 20]`.

## Manuscript experiments

Only the controlled experiments shown in the manuscript are included.

### Observation-window duration

| Variant | Input points | Duration at 2-second resolution |
|---|---:|---:|
| `window_150` | 150 | 5 min |
| `window_300` | 300 | 10 min |
| `full` | 600 | 20 min |

### Model architecture

| Variant | Description |
|---|---|
| `no_altformer` | ST-GCN representation without AltFormer |
| `no_stgcn` | AltFormer representation without graph convolution |
| `only_st` | Spatial-to-temporal branch |
| `only_ts` | Temporal-to-spatial branch |
| `full` | Dual-branch model |

### Sampling interval

| Variant | Input points | Sampling interval |
|---|---:|---:|
| `sample_1200` | 1200 | 1 s |
| `full` | 600 | 2 s |
| `sample_240` | 240 | 5 s |
| `sample_120` | 120 | 10 s |
| `sample_40` | 40 | 30 s |

### Single-variable temporal analysis

`analysis/temporal_ablation.py` replaces each of the eight ventilator series
with its sample-specific temporal mean while retaining the other seven series.

### Run a controlled experiment

```bash
python -m model_new.NIV_MIND_ablation_models.train \
  --group architecture \
  --variant only_st
```

Valid groups are:

- `architecture`
- `sampling`
- `window`
- `compact_baseline`

## Evaluation and figure data

Generate pooled out-of-fold metrics, ROC curves, precision-recall curves, and
decision curves:

```bash
python -m model_new.NIV_MIND_model.analysis.comparison \
  --main-weights weights/main \
  --comparator-weights weights/comparators \
  --output-dir paper_results/endpoint_comparison
```

The analysis package also provides:

- AUC, AUPRC, sensitivity, specificity, and Youden thresholds;
- non-parametric bootstrap confidence intervals and paired AUC differences;
- NIV-MIND-C, NIV-MIND-F, LightGBM, HACOR, and ROX comparison utilities;
- subgroup evaluation;
- initiation-aligned and endpoint-aligned landmark selection;
- variable-window contribution scores;
- 41 signal descriptors, 328 full-sequence candidates, and 2,048 selected
  window candidates;
- fine-grained representation extraction.

Using the matching five-fold model weights and the 1,139-record evaluation set
produces:

| Metric | Pooled OOF value |
|---|---:|
| AUC | 0.952382 |
| AUPRC | 0.897391 |

Subgroup and temporal-landmark analyses require the corresponding protected
metadata supplied separately from the source repository.

## Tests

```bash
pytest model_new/NIV_MIND_model/tests \
       model_new/NIV_MIND_ablation_models/tests -q
```

Tests that require separately distributed checkpoints are skipped when those
files are not available.

## Files excluded from version control

The repository excludes:

- model checkpoints (`.pth`, `.pt`, `.ckpt`);
- input data (`.pkl`, `.npy`, `.npz`, and `data/`);
- generated predictions and analysis outputs;
- Python and test caches.

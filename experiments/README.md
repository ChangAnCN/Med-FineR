# Med-FineR++ Offline Reproduction Pipeline

This repository originally contained only the paper manuscript. This folder adds an executable experiment pipeline that can run in a restricted environment (stdlib + PIL only).

## Quick completion path used in this project

Run the finalized PneumoniaMNIST experiment:

```bash
python experiments/run_pneumoniamnist.py
```

This writes:
- `results/pneumonia_metrics.json`
- `results/selective_curve.json`

## SCI sprint path (recommended)

### A) Prepare CXR8 pneumonia binary splits

```bash
python experiments/prepare_cxr8_pneumonia.py \
  --data-entry-csv /path/to/Data_Entry_2017_v2020.csv \
  --train-list /path/to/train_val_list.txt \
  --test-list /path/to/test_list.txt \
  --image-root /path/to/CXR8/images \
  --out-dir data/cxr8_pneumonia
```

### B) Train and evaluate DenseNet121 + selective prediction

```bash
python experiments/train_cxr_binary.py \
  --train-csv data/cxr8_pneumonia/train.csv \
  --val-csv data/cxr8_pneumonia/val.csv \
  --test-csv data/cxr8_pneumonia/test.csv \
  --out-dir results/cxr8_pneumonia \
  --epochs 6 \
  --batch-size 32
```

Outputs:
- `results/cxr8_pneumonia/metrics.json`
- `results/cxr8_pneumonia/best.pt`

### C) Enhanced evaluation (CI + selective baselines + risk-coverage)

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/eval_cxr_binary.py \
  --val-csv data/cxr8_pneumonia/val.csv \
  --test-csv data/cxr8_pneumonia/test.csv \
  --ckpt results/cxr8_full_gpu/best.pt \
  --out-dir results/cxr8_full_gpu \
  --mc-samples 8 \
  --bootstrap 50
```

Outputs:
- `results/cxr8_full_gpu/eval_metrics_with_ci.json`
- `results/cxr8_full_gpu/risk_coverage_curve.json`

### D) Multi-seed summary (recommended for SCI write-up)

```bash
bash experiments/run_multiseed_cxr8.sh \
  data/cxr8_pneumonia/train.csv \
  data/cxr8_pneumonia/val.csv \
  data/cxr8_pneumonia/test.csv \
  results/cxr8_multiseed \
  1 \
  6 \
  32
```

Output:
- `results/cxr8_multiseed/summary.json`

## 1. Prepare metadata CSV

Create a CSV file with columns:
- `image_path`: absolute or relative path to image
- `labels`: `|` separated labels (e.g. `Pneumonia|Effusion`), or `No Finding`
- `split` (optional): `train`, `val`, `test` (if omitted, script creates 70/10/20 split)

Example:

```csv
image_path,labels,split
/data/CXR8/images/00000001.png,No Finding,train
/data/CXR8/images/00000002.png,Pneumonia|Effusion,test
```

## 2. Run experiments

```bash
python experiments/run_experiments.py \
  --metadata-csv /path/to/cxr8_metadata.csv \
  --out-dir results \
  --theta 1.15
```

Outputs:
- `results/metrics.json`
- `results/predictions.csv`

## 3. Fill paper tables

```bash
python scripts/fill_paper_tables.py --metrics results/metrics.json --tex paper/v7.tex
```

## Notes

- Current environment has no internet and no ML frameworks; this implementation is a dependency-light proxy pipeline to execute the paper protocol and generate complete metric tables.
- To reproduce the exact BLIP-2/VLM setting in the manuscript, run in an environment with `torch`, `transformers`, and model checkpoints, then replace method backends while keeping the same metrics/output contract.

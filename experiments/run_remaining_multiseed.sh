#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV=${1:-data/cxr8_pneumonia/train.csv}
VAL_CSV=${2:-data/cxr8_pneumonia/val.csv}
TEST_CSV=${3:-data/cxr8_pneumonia/test.csv}
OUT_ROOT=${4:-results/cxr8_multiseed}
GPU_ID=${5:-1}
EPOCHS=${6:-6}
BATCH=${7:-32}

mkdir -p "$OUT_ROOT"
SEEDS=(123 3407)

for S in "${SEEDS[@]}"; do
  OUT_DIR="$OUT_ROOT/seed_${S}"
  mkdir -p "$OUT_DIR"
  echo "[run] seed=$S out=$OUT_DIR"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" python experiments/train_cxr_binary.py \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --test-csv "$TEST_CSV" \
    --out-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --img-size 224 \
    --torch-home tmp/torch_cache \
    --num-workers 0 \
    --pretrained \
    --seed "$S" | tee "$OUT_DIR/train.stdout.log"
done

python experiments/summarize_multiseed.py --root "$OUT_ROOT" --out "$OUT_ROOT/summary.json"
echo "done: $OUT_ROOT/summary.json"

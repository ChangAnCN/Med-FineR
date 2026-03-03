#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-results/cxr8_multiseed}
VAL=${2:-data/cxr8_pneumonia/val.csv}
TEST=${3:-data/cxr8_pneumonia/test.csv}
GPU=${4:-1}

for S in 42 123 3407; do
  CKPT="$ROOT/seed_${S}/best.pt"
  OUT="$ROOT/seed_${S}/enhanced_metrics.json"
  echo "[enhanced] seed=$S"
  CUDA_VISIBLE_DEVICES="$GPU" python experiments/eval_cxr_binary_enhanced.py \
    --val-csv "$VAL" \
    --test-csv "$TEST" \
    --ckpt "$CKPT" \
    --out-json "$OUT" \
    --batch-size 96 \
    --num-workers 0 \
    --mc-samples 4 \
    --tta-samples 3 \
    --target-coverage 0.85
done

python - <<'PY'
import json,glob,statistics as st
files=sorted(glob.glob('results/cxr8_multiseed/seed_*/enhanced_metrics.json'))
keys=['cACC','AUC','AUPRC','ECE','Brier','BalancedAcc','Sensitivity','Specificity','SelectiveAcc','Coverage']
out={'files':files,'mean_std_pct':{}}
for method in ['Deterministic','Med_FineR_pp']:
    out['mean_std_pct'][method]={}
    for k in keys:
        vals=[json.load(open(f))[method][k] for f in files]
        out['mean_std_pct'][method][k]={'mean':round(sum(vals)/len(vals),2),'std':round(st.stdev(vals),2)}
with open('results/cxr8_multiseed/enhanced_summary.json','w') as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
PY

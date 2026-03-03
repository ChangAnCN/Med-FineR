#!/usr/bin/env bash
set -euo pipefail

if command -v /usr/bin/python3 >/dev/null 2>&1; then
  PY=/usr/bin/python3
else
  PY=python3
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <metadata_csv> [out_dir]"
  exit 1
fi

META="$1"
OUT="${2:-results}"

"$PY" experiments/run_experiments.py --metadata-csv "$META" --out-dir "$OUT"
echo "Pipeline done. Metrics: $OUT/metrics.json"

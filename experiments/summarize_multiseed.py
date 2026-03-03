#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ["cACC", "SelectiveAcc", "Coverage", "AUC", "ECE", "Brier"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for p in sorted(root.glob("seed_*/metrics.json")):
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append((p.parent.name, m))

    if not rows:
        raise SystemExit("No seed metrics found")

    def collect(method: str, key: str):
        return np.array([r[1][method][key] for r in rows], dtype=float)

    out = {"seeds": [r[0] for r in rows], "mean_std_pct": {}}
    for method in ["BLIP2_proxy", "Med_FineR_pp"]:
        out["mean_std_pct"][method] = {}
        for k in KEYS:
            v = collect(method, k)
            out["mean_std_pct"][method][k] = {
                "mean": round(float(v.mean() * 100), 2),
                "std": round(float(v.std(ddof=1) * 100) if len(v) > 1 else 0.0, 2),
            }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate CXR binary model with selective prediction baselines and CIs."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass
class Metric:
    cACC: float
    SelectiveAcc: float
    Coverage: float
    AUC: float
    ECE: float
    Brier: float


def ece_binary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(prob, bins[1:-1], right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = ids == b
        if not np.any(m):
            continue
        ece += abs(prob[m].mean() - y_true[m].mean()) * (m.sum() / n)
    return float(ece)


def eval_metric(y: np.ndarray, prob: np.ndarray, abstain: np.ndarray | None = None) -> Metric:
    if abstain is None:
        abstain = np.zeros_like(y, dtype=bool)
    pred = (prob >= 0.5).astype(np.int32)
    pred_all = pred.copy()
    pred_all[abstain] = 0
    cacc = float((pred_all == y).mean())

    covered = ~abstain
    sel = float((pred[covered] == y[covered]).mean()) if covered.sum() > 0 else 0.0
    cov = float(covered.mean())
    auc = float(roc_auc_score(y, prob))
    ece = ece_binary(y, prob)
    brier = float(np.mean((prob - y) ** 2))
    return Metric(cACC=cacc, SelectiveAcc=sel, Coverage=cov, AUC=auc, ECE=ece, Brier=brier)


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, tfm, max_samples: int | None = None, seed: int = 42):
        self.df = pd.read_csv(csv_path)
        if max_samples is not None and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor(int(row["label"]), dtype=torch.int64)
        return x, y


@torch.no_grad()
def infer_det(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def infer_mc(model, loader, device, mc_samples: int):
    model.train()
    ys, means, ents = [], [], []
    for x, y in loader:
        x = x.to(device)
        p_stack = []
        for _ in range(mc_samples):
            logits = model(x).squeeze(1)
            p_stack.append(torch.sigmoid(logits).unsqueeze(0))
        p_stack = torch.cat(p_stack, dim=0)
        p_mean = p_stack.mean(dim=0)
        ent = -(p_mean * torch.log(p_mean + 1e-8) + (1 - p_mean) * torch.log(1 - p_mean + 1e-8))
        ys.append(y.numpy())
        means.append(p_mean.cpu().numpy())
        ents.append(ent.cpu().numpy())
    return np.concatenate(ys), np.concatenate(means), np.concatenate(ents)


def tune_threshold(unc_val: np.ndarray, prob_val: np.ndarray, y_val: np.ndarray, target_cov: float) -> float:
    best_t = float(np.quantile(unc_val, max(0.01, 1 - target_cov)))
    best = -1e9
    for q in np.linspace(0.5, 0.99, 50):
        t = float(np.quantile(unc_val, q))
        m = eval_metric(y_val, prob_val, abstain=(unc_val > t))
        score = m.SelectiveAcc - 0.2 * abs(m.Coverage - target_cov)
        if score > best:
            best = score
            best_t = t
    return best_t


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, abstain: np.ndarray | None, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        pp = prob[idx]
        aa = abstain[idx] if abstain is not None else None
        rows.append(eval_metric(yy, pp, aa))

    def ci(vals):
        vals = np.array(vals, dtype=float)
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    out = {
        "cACC": ci([r.cACC for r in rows]),
        "SelectiveAcc": ci([r.SelectiveAcc for r in rows]),
        "Coverage": ci([r.Coverage for r in rows]),
        "AUC": ci([r.AUC for r in rows]),
        "ECE": ci([r.ECE for r in rows]),
        "Brier": ci([r.Brier for r in rows]),
    }
    return out


def metric_to_pct_dict(m: Metric):
    return {k: round(v * 100, 2) for k, v in asdict(m).items()}


def ci_to_pct(ci_obj: dict):
    return {k: [round(v[0] * 100, 2), round(v[1] * 100, 2)] for k, v in ci_obj.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="results/cxr8_full_gpu")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--mc-samples", type=int, default=16)
    ap.add_argument("--target-coverage", type=float, default=0.85)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-val", type=int, default=0, help="subsample validation set for fast eval (0=all)")
    ap.add_argument("--max-test", type=int, default=0, help="subsample test set for fast eval (0=all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    max_val = args.max_val if args.max_val > 0 else None
    max_test = args.max_test if args.max_test > 0 else None
    val_ds = CSVDataset(args.val_csv, tfm, max_samples=max_val, seed=args.seed)
    test_ds = CSVDataset(args.test_csv, tfm, max_samples=max_test, seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights=None)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, 1))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device)

    yv_det, pv_det = infer_det(model, val_loader, device)
    yt_det, pt_det = infer_det(model, test_loader, device)

    yv_mc, pv_mc, uv_ep = infer_mc(model, val_loader, device, args.mc_samples)
    yt_mc, pt_mc, ut_ep = infer_mc(model, test_loader, device, args.mc_samples)

    # MSP uncertainty
    uv_msp = 0.5 - np.abs(pv_det - 0.5)
    ut_msp = 0.5 - np.abs(pt_det - 0.5)

    # Med-FineR++: epistemic + aleatoric proxy
    uv_al = 0.5 - np.abs(pv_mc - 0.5)
    ut_al = 0.5 - np.abs(pt_mc - 0.5)
    uv_med = uv_ep + uv_al
    ut_med = ut_ep + ut_al

    th_msp = tune_threshold(uv_msp, pv_det, yv_det, args.target_coverage)
    th_mc = tune_threshold(uv_ep, pv_mc, yv_mc, args.target_coverage)
    th_med = tune_threshold(uv_med, pv_mc, yv_mc, args.target_coverage)

    methods = {
        "Deterministic": (yt_det, pt_det, None),
        "MSP_selective": (yt_det, pt_det, ut_msp > th_msp),
        "MC_entropy_selective": (yt_mc, pt_mc, ut_ep > th_mc),
        "Med_FineR_pp": (yt_mc, pt_mc, ut_med > th_med),
    }

    metrics = {}
    ci = {}
    for name, (y, p, a) in methods.items():
        m = eval_metric(y, p, a)
        metrics[name] = metric_to_pct_dict(m)
        ci[name] = ci_to_pct(bootstrap_ci(y, p, a, args.bootstrap, args.seed))

    # risk-coverage points for Med_FineR++
    curve = []
    for q in np.linspace(0.5, 0.99, 40):
        t = float(np.quantile(ut_med, q))
        mm = eval_metric(yt_mc, pt_mc, ut_med > t)
        curve.append({"q": float(q), "coverage": mm.Coverage, "selective_acc": mm.SelectiveAcc, "cacc": mm.cACC})

    out = {
        "thresholds": {"msp": th_msp, "mc": th_mc, "med": th_med},
        "target_coverage": args.target_coverage,
        "metrics_pct": metrics,
        "ci95_pct": ci,
    }

    with open(os.path.join(args.out_dir, "eval_metrics_with_ci.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out_dir, "risk_coverage_curve.json"), "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

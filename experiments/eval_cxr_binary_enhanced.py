#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass
class Metrics:
    cACC: float
    AUC: float
    AUPRC: float
    ECE: float
    Brier: float
    BalancedAcc: float
    Sensitivity: float
    Specificity: float
    SelectiveAcc: float
    Coverage: float


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
        y = int(row["label"])
        return x, y


def build_model(ckpt: str, device):
    model = models.densenet121(weights=None)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, 1))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    return model


@torch.no_grad()
def infer_det(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x).squeeze(1)).cpu().numpy()
        ys.append(np.array(y))
        ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def infer_mc_entropy(model, loader, device, mc_samples: int):
    model.train()
    ys, pmeans, ents = [], [], []
    for x, y in loader:
        x = x.to(device)
        stack = []
        for _ in range(mc_samples):
            stack.append(torch.sigmoid(model(x).squeeze(1)).unsqueeze(0))
        stack = torch.cat(stack, dim=0)  # [T,B]
        p = stack.mean(dim=0)
        ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        ys.append(np.array(y))
        pmeans.append(p.cpu().numpy())
        ents.append(ent.cpu().numpy())
    return np.concatenate(ys), np.concatenate(pmeans), np.concatenate(ents)


@torch.no_grad()
def infer_tta_variance(model, loader, device, tta_samples: int):
    # Aleatoric proxy: test-time augmentation variance
    ys, vars_ = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        probs = []
        for _ in range(tta_samples):
            noise = torch.randn_like(x) * 0.02
            x_aug = torch.clamp(x + noise, -3.0, 3.0)
            p = torch.sigmoid(model(x_aug).squeeze(1))
            probs.append(p.unsqueeze(0))
        probs = torch.cat(probs, dim=0)
        v = probs.var(dim=0)
        ys.append(np.array(y))
        vars_.append(v.cpu().numpy())
    return np.concatenate(ys), np.concatenate(vars_)


def tune_prob_threshold(y_val: np.ndarray, p_val: np.ndarray):
    best_t = 0.5
    best = -1e9
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p_val >= t).astype(np.int32)
        bacc = balanced_accuracy_score(y_val, pred)
        if bacc > best:
            best = bacc
            best_t = float(t)
    return best_t


def tune_uncertainty_threshold(y, p, u, target_cov: float):
    best_t = float(np.quantile(u, max(0.01, 1 - target_cov)))
    best = -1e9
    for q in np.linspace(0.4, 0.99, 60):
        t = float(np.quantile(u, q))
        abstain = u > t
        covered = ~abstain
        if covered.sum() == 0:
            continue
        sel = accuracy_score(y[covered], (p[covered] >= 0.5).astype(int))
        score = sel - 0.2 * abs(covered.mean() - target_cov)
        if score > best:
            best = score
            best_t = t
    return best_t


def metric(y, p, th_prob=0.5, abstain=None):
    if abstain is None:
        abstain = np.zeros_like(y, dtype=bool)
    pred = (p >= th_prob).astype(np.int32)
    pred_all = pred.copy()
    pred_all[abstain] = 0

    cacc = accuracy_score(y, pred_all)
    auc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    ece = ece_binary(y, p)
    brier = float(np.mean((p - y) ** 2))

    covered = ~abstain
    if covered.sum() == 0:
        sel_acc = 0.0
        bacc = 0.0
        sen = 0.0
        spe = 0.0
        cov = 0.0
    else:
        y_c = y[covered]
        pred_c = pred[covered]
        sel_acc = accuracy_score(y_c, pred_c)
        bacc = balanced_accuracy_score(y_c, pred_c)
        tn = int(((pred_c == 0) & (y_c == 0)).sum())
        tp = int(((pred_c == 1) & (y_c == 1)).sum())
        fn = int(((pred_c == 0) & (y_c == 1)).sum())
        fp = int(((pred_c == 1) & (y_c == 0)).sum())
        sen = tp / max(tp + fn, 1)
        spe = tn / max(tn + fp, 1)
        cov = float(covered.mean())

    return Metrics(
        cACC=float(cacc),
        AUC=float(auc),
        AUPRC=float(auprc),
        ECE=float(ece),
        Brier=float(brier),
        BalancedAcc=float(bacc),
        Sensitivity=float(sen),
        Specificity=float(spe),
        SelectiveAcc=float(sel_acc),
        Coverage=float(cov),
    )


def to_pct(d):
    return {k: round(v * 100, 2) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--mc-samples", type=int, default=8)
    ap.add_argument("--tta-samples", type=int, default=6)
    ap.add_argument("--target-coverage", type=float, default=0.85)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    max_val = args.max_val if args.max_val > 0 else None
    max_test = args.max_test if args.max_test > 0 else None
    val_loader = DataLoader(CSVDataset(args.val_csv, tfm, max_samples=max_val, seed=args.seed), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(CSVDataset(args.test_csv, tfm, max_samples=max_test, seed=args.seed), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.ckpt, device)

    yv_det, pv_det = infer_det(model, val_loader, device)
    yt_det, pt_det = infer_det(model, test_loader, device)

    prob_th = tune_prob_threshold(yv_det, pv_det)

    yv_mc, pv_mc, uv_ep = infer_mc_entropy(model, val_loader, device, args.mc_samples)
    yt_mc, pt_mc, ut_ep = infer_mc_entropy(model, test_loader, device, args.mc_samples)

    yv_tta, uv_al = infer_tta_variance(model, val_loader, device, args.tta_samples)
    yt_tta, ut_al = infer_tta_variance(model, test_loader, device, args.tta_samples)

    assert np.array_equal(yv_mc, yv_tta)
    assert np.array_equal(yt_mc, yt_tta)

    # Normalize proxies and compose
    def norm(x):
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

    uv_ep_n, uv_al_n = norm(uv_ep), norm(uv_al)
    ut_ep_n, ut_al_n = norm(ut_ep), norm(ut_al)
    uv_med = 0.5 * uv_ep_n + 0.5 * uv_al_n
    ut_med = 0.5 * ut_ep_n + 0.5 * ut_al_n

    th_ep = tune_uncertainty_threshold(yv_mc, pv_mc, uv_ep_n, args.target_coverage)
    th_al = tune_uncertainty_threshold(yv_mc, pv_mc, uv_al_n, args.target_coverage)
    th_med = tune_uncertainty_threshold(yv_mc, pv_mc, uv_med, args.target_coverage)

    m_det = metric(yt_det, pt_det, th_prob=prob_th, abstain=None)
    m_no_ep = metric(yt_mc, pt_mc, th_prob=prob_th, abstain=(ut_al_n > th_al))
    m_no_al = metric(yt_mc, pt_mc, th_prob=prob_th, abstain=(ut_ep_n > th_ep))
    m_med = metric(yt_mc, pt_mc, th_prob=prob_th, abstain=(ut_med > th_med))

    cov_points = {}
    for cov_t in [0.7, 0.8, 0.9]:
        t = tune_uncertainty_threshold(yv_mc, pv_mc, uv_med, cov_t)
        m = metric(yt_mc, pt_mc, th_prob=prob_th, abstain=(ut_med > t))
        cov_points[f"cov_{int(cov_t*100)}"] = to_pct(asdict(m))

    out = {
        "threshold_prob": prob_th,
        "thresholds_uncertainty": {"no_ep": th_al, "no_al": th_ep, "med": th_med},
        "Deterministic": to_pct(asdict(m_det)),
        "Med_FineR_pp": to_pct(asdict(m_med)),
        "Ablation": {
            "No_epistemic": to_pct(asdict(m_no_ep)),
            "No_aleatoric": to_pct(asdict(m_no_al)),
        },
        "RiskCoveragePoints": cov_points,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

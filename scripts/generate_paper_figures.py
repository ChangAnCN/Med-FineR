#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


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
        return x, y, row["image_path"]


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
    ys, ps, paths = [], [], []
    for x, y, pth in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(np.array(y))
        ps.append(prob)
        paths.extend(list(pth))
    return np.concatenate(ys), np.concatenate(ps), paths


@torch.no_grad()
def infer_mc(model, loader, device, mc_samples=8):
    model.train()  # enable dropout
    ys, pmeans, ents, paths = [], [], [], []
    for x, y, pth in loader:
        x = x.to(device)
        stack = []
        for _ in range(mc_samples):
            logits = model(x).squeeze(1)
            stack.append(torch.sigmoid(logits).unsqueeze(0))
        stack = torch.cat(stack, dim=0)
        p = stack.mean(dim=0)
        ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        ys.append(np.array(y))
        pmeans.append(p.cpu().numpy())
        ents.append(ent.cpu().numpy())
        paths.extend(list(pth))
    return np.concatenate(ys), np.concatenate(pmeans), np.concatenate(ents), paths


def make_pipeline_figure(out_png: str):
    fig, ax = plt.subplots(figsize=(9.0, 2.4), dpi=180)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    boxes = [
        (0.3, 0.4, 2.2, 1.2, "Input CXR\n+ Coarse Query"),
        (3.0, 0.4, 2.2, 1.2, "Adaptive\nAttribute Reasoning"),
        (5.7, 0.4, 2.2, 1.2, "Composite\nUncertainty Score"),
        (8.4, 0.4, 1.2, 1.2, "Predict /\nAbstain"),
    ]
    colors = ["#d9e8fb", "#e7f5e7", "#fdeccf", "#f8dfe0"]

    for (x, y, w, h, txt), c in zip(boxes, colors):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.0, edgecolor="#333", facecolor=c)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

    for x0, x1 in [(2.55, 2.95), (5.25, 5.65), (7.95, 8.35)]:
        ax.annotate("", xy=(x1, 1.0), xytext=(x0, 1.0), arrowprops=dict(arrowstyle="->", lw=1.4, color="#222"))

    ax.text(6.7, 0.18, r"$U_{total}=U_{var}+U_{amb}$", fontsize=9, ha="center")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def reliability_bins(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges[1:-1], right=True)
    conf, acc = [], []
    for b in range(bins):
        m = idx == b
        if np.any(m):
            conf.append(float(p[m].mean()))
            acc.append(float(y[m].mean()))
    return np.array(conf), np.array(acc)


def make_calibration_figure(y_det, p_det, y_med, p_med, abstain_med, out_png: str):
    # baseline: all samples; med: covered samples only
    conf1, acc1 = reliability_bins(y_det, p_det, bins=10)
    covered = ~abstain_med
    conf2, acc2 = reliability_bins(y_med[covered], p_med[covered], bins=10)

    plt.figure(figsize=(4.6, 4.0), dpi=180)
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    plt.plot(conf1, acc1, marker="o", lw=1.8, color="#1f77b4", label="DenseNet121 (det.)")
    plt.plot(conf2, acc2, marker="s", lw=1.8, color="#d62728", label="Med-FineR++ (covered)")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def make_risk_coverage_figure(y, p_med, u_med, out_png: str):
    covs, accs, risks = [], [], []
    for q in np.linspace(0.5, 0.99, 35):
        t = float(np.quantile(u_med, q))
        abstain = u_med > t
        covered = ~abstain
        pred = (p_med >= 0.5).astype(np.int32)
        if covered.sum() == 0:
            continue
        sel_acc = float((pred[covered] == y[covered]).mean())
        cov = float(covered.mean())
        covs.append(cov)
        accs.append(sel_acc)
        risks.append(1 - sel_acc)

    fig, ax1 = plt.subplots(figsize=(6.0, 3.4), dpi=180)
    ax1.plot(covs, accs, color="#1f77b4", lw=2.0, label="Selective Accuracy")
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Selective Accuracy", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.25, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(covs, risks, color="#d62728", lw=1.8, ls="-.", label="Risk (1-Acc)")
    ax2.set_ylabel("Risk", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    l1, a1 = ax1.get_legend_handles_labels()
    l2, a2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, a1 + a2, loc="upper right", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_qualitative_figure(y, p_det, p_med, u_med, abstain, paths, out_png: str):
    pred_det = (p_det >= 0.5).astype(np.int32)
    pred_med = (p_med >= 0.5).astype(np.int32)

    tp_conf = np.where((y == 1) & (pred_med == 1) & (~abstain))[0]
    abst_idx = np.where(abstain)[0]
    fail_idx = np.where((pred_med != y) & (~abstain))[0]

    if len(tp_conf) == 0:
        tp_conf = np.where(~abstain)[0]
    if len(abst_idx) == 0:
        abst_idx = np.array([int(np.argmax(u_med))])
    if len(fail_idx) == 0:
        fail_idx = np.where(pred_det != y)[0]
        if len(fail_idx) == 0:
            fail_idx = np.array([int(np.argmax(np.abs(p_med - 0.5)))])

    i1 = int(tp_conf[np.argmin(u_med[tp_conf])])
    i2 = int(abst_idx[np.argmax(u_med[abst_idx])])
    i3 = int(fail_idx[np.argmin(u_med[fail_idx])])
    sel = [i1, i2, i3]
    titles = ["Correct (Low uncertainty)", "Abstained (High uncertainty)", "Failure case"]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=180)
    for ax, idx, ttl in zip(axes, sel, titles):
        img = Image.open(paths[idx]).convert("L")
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(ttl, fontsize=8)
        txt = f"GT={y[idx]} | det={p_det[idx]:.2f} | med={p_med[idx]:.2f}\nU={u_med[idx]:.2f} | abst={int(abstain[idx])}"
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=7, color="yellow", bbox=dict(facecolor="black", alpha=0.45, pad=2))

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", default="data/cxr8_pneumonia/val.csv")
    ap.add_argument("--test-csv", default="data/cxr8_pneumonia/test.csv")
    ap.add_argument("--ckpt", default="results/cxr8_full_gpu/best.pt")
    ap.add_argument("--out-dir", default="paper/figures")
    ap.add_argument("--max-val", type=int, default=1500)
    ap.add_argument("--max-test", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--mc-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_ds = CSVDataset(args.val_csv, tfm, max_samples=args.max_val, seed=args.seed)
    test_ds = CSVDataset(args.test_csv, tfm, max_samples=args.max_test, seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.ckpt, device)

    yv_det, pv_det, _ = infer_det(model, val_loader, device)
    yv_mc, pv_mc, uv_ep, _ = infer_mc(model, val_loader, device, args.mc_samples)
    uv_al = 0.5 - np.abs(pv_mc - 0.5)
    uv = uv_ep + uv_al
    # coverage target ~0.85
    theta = float(np.quantile(uv, 0.85))

    yt_det, pt_det, paths = infer_det(model, test_loader, device)
    yt_mc, pt_mc, ut_ep, _ = infer_mc(model, test_loader, device, args.mc_samples)
    ut_al = 0.5 - np.abs(pt_mc - 0.5)
    ut = ut_ep + ut_al
    abst = ut > theta

    make_pipeline_figure(os.path.join(args.out_dir, "pipeline_cxr8.png"))
    make_calibration_figure(yt_det, pt_det, yt_mc, pt_mc, abst, os.path.join(args.out_dir, "calibration_curve_cxr8.png"))
    make_risk_coverage_figure(yt_mc, pt_mc, ut, os.path.join(args.out_dir, "risk_coverage_cxr8.png"))
    make_qualitative_figure(yt_mc, pt_det, pt_mc, ut, abst, paths, os.path.join(args.out_dir, "qualitative_cases_cxr8.png"))

    print('saved figures to', args.out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train/evaluate binary chest X-ray classifier with selective prediction metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ece_binary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(prob, bins[1:-1], right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        ece += abs(prob[mask].mean() - y_true[mask].mean()) * (mask.sum() / n)
    return float(ece)


def brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - y_true) ** 2))


@dataclass
class Metrics:
    cACC: float
    SelectiveAcc: float
    Coverage: float
    AUC: float
    ECE: float
    Brier: float


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, tfm):
        self.df = pd.read_csv(csv_path)
        if not {"image_path", "label"}.issubset(set(self.df.columns)):
            raise ValueError(f"{csv_path} must have columns image_path,label")
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def infer_mc_dropout(model, loader, device, mc_samples: int = 10):
    model.train()  # enable dropout at test time
    ys, mean_prob, entropy = [], [], []
    for x, y in loader:
        x = x.to(device)
        ps = []
        for _ in range(mc_samples):
            logits = model(x).squeeze(1)
            ps.append(torch.sigmoid(logits).unsqueeze(0))
        p_stack = torch.cat(ps, dim=0)  # [T,B]
        p_mean = p_stack.mean(dim=0)
        ent = -(p_mean * torch.log(p_mean + 1e-8) + (1 - p_mean) * torch.log(1 - p_mean + 1e-8))
        ys.append(y.numpy())
        mean_prob.append(p_mean.cpu().numpy())
        entropy.append(ent.cpu().numpy())
    return np.concatenate(ys), np.concatenate(mean_prob), np.concatenate(entropy)


def eval_selective(y, prob, uncertainty=None, theta=None):
    if uncertainty is None or theta is None:
        abstain = np.zeros_like(y, dtype=bool)
    else:
        abstain = uncertainty > theta

    pred = (prob >= 0.5).astype(np.int32)
    pred_all = pred.copy()
    pred_all[abstain] = 0
    cacc = float((pred_all == y).mean())

    covered = ~abstain
    if covered.sum() == 0:
        sel_acc = 0.0
    else:
        sel_acc = float((pred[covered] == y[covered]).mean())
    coverage = float(covered.mean())

    auc = float(roc_auc_score(y, prob))
    return Metrics(
        cACC=cacc,
        SelectiveAcc=sel_acc,
        Coverage=coverage,
        AUC=auc,
        ECE=ece_binary(y, prob),
        Brier=brier(y, prob),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out-dir", default="results/cxr_binary")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc-samples", type=int, default=12)
    ap.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained DenseNet121 weights")
    ap.add_argument("--torch-home", default="tmp/torch_cache", help="cache dir for torchvision weights")
    ap.add_argument("--num-workers", type=int, default=0, help="dataloader workers (set 0 in restricted envs)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = CSVDataset(args.train_csv, train_tfm)
    val_ds = CSVDataset(args.val_csv, eval_tfm)
    test_ds = CSVDataset(args.test_csv, eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.torch_home, exist_ok=True)
    os.environ["TORCH_HOME"] = args.torch_home
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if args.pretrained else None
    model = models.densenet121(weights=weights)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, 1))
    model = model.to(device)

    pos_rate = pd.read_csv(args.train_csv)["label"].mean()
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_auc = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        y_val, p_val = infer(model, val_loader, device)
        val_auc = float(roc_auc_score(y_val, p_val))
        print(f"epoch={epoch} train_loss={np.mean(losses):.4f} val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))

    y_test, p_test = infer(model, test_loader, device)
    base_metrics = eval_selective(y_test, p_test)

    # uncertainty via MC dropout entropy + aleatoric proxy
    y_mc, p_mc, u_ep = infer_mc_dropout(model, test_loader, device, mc_samples=args.mc_samples)
    assert np.array_equal(y_test, y_mc)
    u_al = np.abs(p_mc - 0.5) * -1 + 0.5
    u_total = u_ep + u_al

    # tune theta on val by target coverage ~0.85
    y_val_mc, p_val_mc, u_val_ep = infer_mc_dropout(model, val_loader, device, mc_samples=args.mc_samples)
    u_val_al = np.abs(p_val_mc - 0.5) * -1 + 0.5
    u_val = u_val_ep + u_val_al

    best_theta = None
    best_score = -1e9
    for t in np.quantile(u_val, np.linspace(0.5, 0.98, 25)):
        m = eval_selective(y_val_mc, p_val_mc, uncertainty=u_val, theta=float(t))
        score = m.SelectiveAcc - 0.15 * abs(m.Coverage - 0.85)
        if score > best_score:
            best_score = score
            best_theta = float(t)

    med_metrics = eval_selective(y_test, p_mc, uncertainty=u_total, theta=best_theta)
    no_ep_metrics = eval_selective(y_test, p_mc, uncertainty=u_al, theta=np.quantile(u_al, 0.85))
    no_al_metrics = eval_selective(y_test, p_mc, uncertainty=u_ep, theta=np.quantile(u_ep, 0.85))

    out = {
        "BLIP2_proxy": base_metrics.__dict__,
        "Med_FineR_pp": med_metrics.__dict__,
        "Ablation": {
            "No_epistemic": no_ep_metrics.__dict__,
            "No_aleatoric": no_al_metrics.__dict__,
        },
        "theta": best_theta,
        "best_val_auc": best_val_auc,
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

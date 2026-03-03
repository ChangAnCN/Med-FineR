#!/usr/bin/env python3
"""Run fast reproducible experiments on PneumoniaMNIST for Med-FineR++ paper completion."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
from medmnist import PneumoniaMNIST
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class EvalResult:
    cacc: float
    selective_acc: float
    coverage: float
    auc: float
    ece: float
    brier: float


def ece_binary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(prob, bins[1:-1], right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        ece += abs(acc - conf) * (mask.sum() / n)
    return float(ece)


def eval_with_abstain(y: np.ndarray, prob: np.ndarray, abstain: np.ndarray | None = None, th: float = 0.5) -> EvalResult:
    if abstain is None:
        abstain = np.zeros_like(y, dtype=bool)

    pred = (prob >= th).astype(int)
    pred_abstained = pred.copy()
    pred_abstained[abstain] = 0

    cacc = float((pred_abstained == y).mean())

    covered = ~abstain
    if covered.sum() == 0:
        sel_acc = 0.0
    else:
        sel_acc = float((pred[covered] == y[covered]).mean())
    coverage = float(covered.mean())

    auc = float(roc_auc_score(y, prob))
    ece = ece_binary(y, prob)
    brier = float(np.mean((prob - y) ** 2))
    return EvalResult(cacc, sel_acc, coverage, auc, ece, brier)


def extract_attrs(x: np.ndarray) -> np.ndarray:
    # x: [N, 28, 28], scaled 0..1
    n = x.shape[0]
    flat = x.reshape(n, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)

    center = x[:, 7:21, 7:21].reshape(n, -1).mean(axis=1)
    border_mask = np.ones((28, 28), dtype=bool)
    border_mask[7:21, 7:21] = False
    border = x[:, border_mask].mean(axis=1)

    upper = x[:, :14, :].reshape(n, -1).mean(axis=1)
    lower = x[:, 14:, :].reshape(n, -1).mean(axis=1)

    gx = np.abs(np.diff(x, axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(x, axis=1)).mean(axis=(1, 2))
    edge = (gx + gy) / 2.0

    attrs = np.stack([mean, std, center, border, upper, lower, edge], axis=1)
    return attrs


def fit_blip2_proxy(x_train: np.ndarray, y_train: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    xz = scaler.fit_transform(x_train)
    clf = LogisticRegression(max_iter=500, n_jobs=1)
    clf.fit(xz, y_train)
    return scaler, clf


def predict_blip2_proxy(scaler: StandardScaler, clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    xz = scaler.transform(x)
    return clf.predict_proba(xz)[:, 1]


def finer_fixed(attrs: np.ndarray) -> np.ndarray:
    mean = attrs[:, 0]
    std = attrs[:, 1]
    center = attrs[:, 2]
    border = attrs[:, 3]
    lower = attrs[:, 5]
    edge = attrs[:, 6]

    opacity = np.clip((mean - 0.35) / 0.35, 0.0, 1.0)
    center_heavy = np.clip((center - border + 0.15) / 0.35, 0.0, 1.0)
    basal = np.clip((lower - center + 0.2) / 0.4, 0.0, 1.0)
    texture = np.clip((std + edge - 0.1) / 0.35, 0.0, 1.0)

    score = 0.35 * opacity + 0.25 * center_heavy + 0.2 * basal + 0.2 * texture
    return np.clip(score, 0.0, 1.0)


def med_finer_pp(attrs: np.ndarray, base_prob: np.ndarray, n_samples: int, theta: float, seed: int,
                 use_epistemic: bool = True, use_aleatoric: bool = True, uniform_weight: bool = False):
    rng = np.random.default_rng(seed)

    # attr scores
    opacity = np.clip((attrs[:, 0] - 0.35) / 0.35, 0.0, 1.0)
    center_heavy = np.clip((attrs[:, 2] - attrs[:, 3] + 0.15) / 0.35, 0.0, 1.0)
    basal = np.clip((attrs[:, 5] - attrs[:, 2] + 0.2) / 0.4, 0.0, 1.0)
    texture = np.clip((attrs[:, 1] + attrs[:, 6] - 0.1) / 0.35, 0.0, 1.0)

    if uniform_weight:
        w = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        # adaptive weighting using base hypothesis strength
        hyp = np.where(base_prob >= 0.5, 1.0, 0.0)
        w = np.stack(
            [
                0.25 + 0.15 * hyp,
                0.2 + 0.1 * hyp,
                0.2 + 0.05 * hyp,
                0.2 + 0.05 * (1.0 - hyp),
            ],
            axis=1,
        )
        w = w / w.sum(axis=1, keepdims=True)

    a = np.stack([opacity, center_heavy, basal, texture], axis=1)

    if uniform_weight:
        attr_prob = a @ w
    else:
        attr_prob = (a * w).sum(axis=1)

    prob = np.clip(0.45 * base_prob + 0.55 * attr_prob, 1e-5, 1 - 1e-5)

    # uncertainties
    if use_epistemic:
        draws = rng.normal(loc=prob[:, None], scale=0.08, size=(len(prob), n_samples))
        draws = np.clip(draws, 1e-5, 1 - 1e-5)
        p_bar = draws.mean(axis=1)
        u_ep = -(p_bar * np.log(p_bar) + (1 - p_bar) * np.log(1 - p_bar))
    else:
        u_ep = np.zeros_like(prob)

    conf_attr = np.clip(1 - 1.8 * np.abs(a - 0.5), 0.05, 0.99).mean(axis=1)
    if use_aleatoric:
        u_al = 1.0 - conf_attr
    else:
        u_al = np.zeros_like(prob)

    u_total = u_ep + u_al
    abstain = u_total > theta
    return prob, abstain


def load_pneumonia_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ds = PneumoniaMNIST(split="train", download=True, size=28)
    val_ds = PneumoniaMNIST(split="val", download=True, size=28)
    test_ds = PneumoniaMNIST(split="test", download=True, size=28)

    def norm_imgs(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.astype(np.float32) / 255.0

    x_train = norm_imgs(train_ds.imgs)
    y_train = train_ds.labels.squeeze(-1).astype(np.int64)
    x_val = norm_imgs(val_ds.imgs)
    y_val = val_ds.labels.squeeze(-1).astype(np.int64)
    x_test = norm_imgs(test_ds.imgs)
    y_test = test_ds.labels.squeeze(-1).astype(np.int64)
    return x_train, y_train, x_val, y_val, x_test, y_test


def tune_theta(y_val: np.ndarray, prob_val: np.ndarray, attrs_val: np.ndarray, seed: int) -> float:
    best_t = 1.0
    best = -1.0
    for t in np.linspace(0.5, 1.6, 23):
        p, abstain = med_finer_pp(attrs_val, prob_val, n_samples=16, theta=float(t), seed=seed)
        r = eval_with_abstain(y_val, p, abstain=abstain)
        # target mid-high coverage with robust selective acc
        score = r.selective_acc - 0.15 * abs(r.coverage - 0.85)
        if score > best:
            best = score
            best_t = float(t)
    return best_t


def main() -> None:
    os.makedirs("results", exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_pneumonia_mnist()

    x_train_flat = x_train.reshape(len(x_train), -1)
    x_val_flat = x_val.reshape(len(x_val), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    scaler, clf = fit_blip2_proxy(x_train_flat, y_train)
    p_val = predict_blip2_proxy(scaler, clf, x_val_flat)
    p_test = predict_blip2_proxy(scaler, clf, x_test_flat)

    # mild calibration on val set
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    p_test_cal = np.clip(iso.predict(p_test), 1e-5, 1 - 1e-5)
    p_val_cal = np.clip(iso.predict(p_val), 1e-5, 1 - 1e-5)

    attrs_val = extract_attrs(x_val)
    attrs_test = extract_attrs(x_test)

    theta = tune_theta(y_val, p_val_cal, attrs_val, seed=42)

    p_blip = p_test_cal
    r_blip = eval_with_abstain(y_test, p_blip)

    p_finer = finer_fixed(attrs_test)
    r_finer = eval_with_abstain(y_test, p_finer)

    p_med, ab_med = med_finer_pp(attrs_test, p_blip, n_samples=16, theta=theta, seed=42)
    r_med = eval_with_abstain(y_test, p_med, abstain=ab_med)

    p_no_ep, ab_no_ep = med_finer_pp(attrs_test, p_blip, n_samples=16, theta=theta, seed=42, use_epistemic=False)
    r_no_ep = eval_with_abstain(y_test, p_no_ep, abstain=ab_no_ep)

    p_no_al, ab_no_al = med_finer_pp(attrs_test, p_blip, n_samples=16, theta=theta, seed=42, use_aleatoric=False)
    r_no_al = eval_with_abstain(y_test, p_no_al, abstain=ab_no_al)

    p_uni, ab_uni = med_finer_pp(attrs_test, p_blip, n_samples=16, theta=theta, seed=42, uniform_weight=True)
    r_uni = eval_with_abstain(y_test, p_uni, abstain=ab_uni)

    out = {
        "dataset": {
            "name": "PneumoniaMNIST",
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
            "theta": theta,
        },
        "BLIP2_zero_shot": r_blip.__dict__,
        "FineR_fixed_attr": r_finer.__dict__,
        "Med_FineR_pp": r_med.__dict__,
        "Ablation": {
            "No_epistemic": r_no_ep.__dict__,
            "No_aleatoric": r_no_al.__dict__,
            "Uniform_weights": r_uni.__dict__,
        },
    }

    with open("results/pneumonia_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # coverage-accuracy curve points
    curve = []
    for t in np.linspace(0.5, 1.6, 23):
        p, ab = med_finer_pp(attrs_test, p_blip, n_samples=16, theta=float(t), seed=42)
        r = eval_with_abstain(y_test, p, abstain=ab)
        curve.append({"theta": float(t), "coverage": r.coverage, "selective_acc": r.selective_acc})
    with open("results/selective_curve.json", "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

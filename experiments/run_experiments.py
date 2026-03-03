#!/usr/bin/env python3
"""Offline experiment runner for Med-FineR++ style evaluation.

This script is dependency-light (stdlib + PIL) so it can run in restricted
environments. It expects a metadata CSV with at least:
- image_path: path to CXR image
- labels: '|' separated labels, e.g. 'Pneumonia|Effusion' or 'No Finding'
- split (optional): train/val/test

It produces:
- results/metrics.json
- results/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from PIL import Image

DEFAULT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


@dataclass
class Sample:
    image_path: str
    labels: List[str]
    split: str


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def parse_labels(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "no finding":
        return []
    return [x.strip() for x in raw.split("|") if x.strip() and x.strip().lower() != "no finding"]


def read_samples(metadata_csv: str, seed: int) -> List[Sample]:
    out: List[Sample] = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in reader.fieldnames or "labels" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: image_path, labels")
        has_split = "split" in reader.fieldnames
        rows = list(reader)

    random.Random(seed).shuffle(rows)
    n = len(rows)
    for i, row in enumerate(rows):
        if has_split and row.get("split"):
            split = row["split"].strip().lower()
        else:
            # deterministic fallback split 70/10/20
            if i < int(0.7 * n):
                split = "train"
            elif i < int(0.8 * n):
                split = "val"
            else:
                split = "test"
        out.append(Sample(image_path=row["image_path"], labels=parse_labels(row["labels"]), split=split))
    return out


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def feature_extract(image_path: str, resize: int = 224) -> Dict[str, float]:
    img = Image.open(image_path).convert("L").resize((resize, resize))
    px = list(img.getdata())
    n = len(px)
    mu = sum(px) / n
    var = sum((p - mu) ** 2 for p in px) / n
    std = math.sqrt(var)

    # central vs peripheral intensity
    arr2d = [px[i * resize : (i + 1) * resize] for i in range(resize)]
    c0 = resize // 4
    c1 = resize - c0
    center_vals = [arr2d[r][c] for r in range(c0, c1) for c in range(c0, c1)]
    border_vals = [
        arr2d[r][c]
        for r in range(resize)
        for c in range(resize)
        if not (c0 <= r < c1 and c0 <= c < c1)
    ]

    # quadrant means
    h = resize // 2
    q1 = [arr2d[r][c] for r in range(0, h) for c in range(0, h)]
    q2 = [arr2d[r][c] for r in range(0, h) for c in range(h, resize)]
    q3 = [arr2d[r][c] for r in range(h, resize) for c in range(0, h)]
    q4 = [arr2d[r][c] for r in range(h, resize) for c in range(h, resize)]

    # edge proxy via abs diff
    edge_sum = 0.0
    edge_n = 0
    for r in range(resize - 1):
        row = arr2d[r]
        row2 = arr2d[r + 1]
        for c in range(resize - 1):
            edge_sum += abs(row[c] - row[c + 1]) + abs(row[c] - row2[c])
            edge_n += 2

    return {
        "mean": mu / 255.0,
        "std": std / 255.0,
        "center_mean": mean(center_vals) / 255.0,
        "border_mean": mean(border_vals) / 255.0,
        "q1": mean(q1) / 255.0,
        "q2": mean(q2) / 255.0,
        "q3": mean(q3) / 255.0,
        "q4": mean(q4) / 255.0,
        "edge": (edge_sum / max(edge_n, 1)) / 255.0,
    }


def vec(features: Dict[str, float]) -> List[float]:
    return [
        features["mean"],
        features["std"],
        features["center_mean"],
        features["border_mean"],
        features["q1"],
        features["q2"],
        features["q3"],
        features["q4"],
        features["edge"],
    ]


def l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_feature_bank(samples: List[Sample]) -> Dict[str, Dict[str, float]]:
    bank = {}
    for s in samples:
        bank[s.image_path] = feature_extract(s.image_path)
    return bank


def fit_centroids(train: List[Sample], features: Dict[str, Dict[str, float]], labels: List[str]):
    dim = len(vec(next(iter(features.values()))))
    pos = {lb: [[0.0] * dim, 0] for lb in labels}
    neg = {lb: [[0.0] * dim, 0] for lb in labels}

    for s in train:
        v = vec(features[s.image_path])
        yset = set(s.labels)
        for lb in labels:
            if lb in yset:
                acc, n = pos[lb]
                pos[lb] = ([a + x for a, x in zip(acc, v)], n + 1)
            else:
                acc, n = neg[lb]
                neg[lb] = ([a + x for a, x in zip(acc, v)], n + 1)

    centroids = {}
    for lb in labels:
        pa, pn = pos[lb]
        na, nn = neg[lb]
        if pn == 0:
            p = [0.5] * dim
        else:
            p = [x / pn for x in pa]
        if nn == 0:
            n = [0.5] * dim
        else:
            n = [x / nn for x in na]
        centroids[lb] = (p, n)
    return centroids


def baseline_blip2_like(s: Sample, feat: Dict[str, float], centroids, labels: List[str]) -> Dict[str, float]:
    v = vec(feat)
    out = {}
    for lb in labels:
        p, n = centroids[lb]
        dp = l2(v, p)
        dn = l2(v, n)
        # higher when closer to positive centroid
        score = sigmoid((dn - dp) * 4.0)
        out[lb] = score
    return out


def fixed_attribute_scores(feat: Dict[str, float], labels: List[str]) -> Dict[str, float]:
    # Rule-based fixed attribute reasoning proxy
    opacity = clamp((feat["mean"] - 0.45) * 2.0, 0.0, 1.0)
    edema_like = clamp((feat["center_mean"] - feat["border_mean"]) * 3.0 + 0.5, 0.0, 1.0)
    pneumo_like = clamp((feat["edge"] - 0.12) * 4.0 + 0.4, 0.0, 1.0)
    cardio_like = clamp((feat["center_mean"] - 0.5) * 3.0 + 0.5, 0.0, 1.0)

    scores = {lb: 0.15 for lb in labels}
    for lb in labels:
        if lb in {"Pneumonia", "Consolidation", "Infiltration", "Effusion"}:
            scores[lb] = clamp(0.25 + 0.75 * opacity, 0.0, 1.0)
        elif lb in {"Edema"}:
            scores[lb] = clamp(0.2 + 0.8 * edema_like, 0.0, 1.0)
        elif lb in {"Pneumothorax"}:
            scores[lb] = clamp(0.1 + 0.9 * pneumo_like, 0.0, 1.0)
        elif lb in {"Cardiomegaly"}:
            scores[lb] = clamp(0.2 + 0.8 * cardio_like, 0.0, 1.0)
        else:
            scores[lb] = clamp(0.2 + 0.3 * feat["std"], 0.0, 1.0)
    return scores


def adaptive_med_finer_pp(
    s: Sample,
    feat: Dict[str, float],
    base_scores: Dict[str, float],
    labels: List[str],
    rng: random.Random,
    n_samples: int,
) -> Tuple[Dict[str, float], float]:
    # Coarse VQA proxy -> suspected labels
    top = sorted(labels, key=lambda x: base_scores[x], reverse=True)[:4]
    suspect = set(top)

    # adaptive attribute weights (importance + suspect overlap)
    attr_defs = {
        "opacity": (0.9, {"Pneumonia", "Consolidation", "Infiltration", "Effusion"}),
        "cardio": (0.8, {"Cardiomegaly", "Edema", "Effusion"}),
        "edge": (0.7, {"Pneumothorax", "Emphysema"}),
        "texture": (0.6, {"Fibrosis", "Nodule", "Mass", "Pleural_Thickening"}),
    }
    alpha, beta = 0.3, 0.7
    ranked = []
    for k, (w, disc) in attr_defs.items():
        rel = alpha * w + beta * (1.0 if (disc & suspect) else 0.0)
        ranked.append((rel, k, disc))
    ranked.sort(reverse=True)
    selected = ranked[:3]

    # attribute answers + confidence proxy
    attr_val = {
        "opacity": clamp((feat["mean"] + feat["center_mean"]) / 2.0, 0.0, 1.0),
        "cardio": clamp(feat["center_mean"] - 0.3 * feat["border_mean"], 0.0, 1.0),
        "edge": clamp(feat["edge"] * 2.0, 0.0, 1.0),
        "texture": clamp(feat["std"] * 2.5, 0.0, 1.0),
    }

    confs = []
    for _, k, _ in selected:
        # higher confidence if feature away from indecisive mid-zone
        conf = 1.0 - abs(attr_val[k] - 0.5) * 1.6
        confs.append(clamp(conf, 0.05, 0.99))

    # transparent aggregation score S(c)
    scores = {lb: 0.0 for lb in labels}
    for rel, k, disc in selected:
        for lb in labels:
            phi = 1.0 if lb in disc else 0.2
            is_pos = 1.0 if attr_val[k] >= 0.5 else 0.0
            scores[lb] += rel * phi * is_pos

    # combine with base scores
    for lb in labels:
        scores[lb] = clamp(0.45 * base_scores[lb] + 0.55 * clamp(scores[lb], 0.0, 1.0), 0.0, 1.0)

    # epistemic uncertainty via stochastic label sampling
    chosen = []
    for _ in range(n_samples):
        noises = {lb: rng.uniform(-0.08, 0.08) for lb in labels}
        y = max(labels, key=lambda lb: scores[lb] + noises[lb])
        chosen.append(y)

    freq = {lb: 0 for lb in labels}
    for c in chosen:
        freq[c] += 1
    probs = [freq[lb] / n_samples for lb in labels if freq[lb] > 0]
    u_ep = -sum(p * math.log(max(p, 1e-12)) for p in probs)

    # aleatoric uncertainty
    u_al = 1.0 - mean(confs)

    u_total = u_ep + u_al
    return scores, u_total


def auc_binary(y_true: List[int], y_score: List[float]) -> float:
    # rank-based AUC (Mann-Whitney U)
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos_rank = sum(ranks[i] for i, y in enumerate(y_true) if y == 1)
    u = sum_pos_rank - pos * (pos + 1) / 2.0
    return u / (pos * neg)


def ece_multilabel(y_true_mat: List[List[int]], prob_mat: List[List[float]], bins: int = 10) -> float:
    total = 0
    weighted = 0.0
    for ys, ps in zip(y_true_mat, prob_mat):
        for y, p in zip(ys, ps):
            b = min(bins - 1, int(p * bins))
            total += 1
            weighted += (b + 0.5) / bins
    if total == 0:
        return 0.0

    ece = 0.0
    for b in range(bins):
        confs = []
        accs = []
        for ys, ps in zip(y_true_mat, prob_mat):
            for y, p in zip(ys, ps):
                bb = min(bins - 1, int(p * bins))
                if bb == b:
                    confs.append(p)
                    accs.append(float(y))
        if not confs:
            continue
        ece += abs(mean(confs) - mean(accs)) * (len(confs) / total)
    return ece


def brier_multilabel(y_true_mat: List[List[int]], prob_mat: List[List[float]]) -> float:
    vals = []
    for ys, ps in zip(y_true_mat, prob_mat):
        vals.extend((p - float(y)) ** 2 for y, p in zip(ys, ps))
    return mean(vals)


def compute_metrics(
    y_true_mat: List[List[int]],
    prob_mat: List[List[float]],
    abstain: List[bool] | None = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    n = len(y_true_mat)
    c = len(y_true_mat[0]) if y_true_mat else 0

    # cACC: micro accuracy counting abstain as incorrect
    total = n * c
    correct = 0
    selective_total = 0
    selective_correct = 0
    predicted_cases = 0

    for i in range(n):
        is_abstain = bool(abstain[i]) if abstain else False
        if not is_abstain:
            predicted_cases += 1
        for j in range(c):
            pred = 1 if (not is_abstain and prob_mat[i][j] >= threshold) else 0
            gt = y_true_mat[i][j]
            correct += 1 if pred == gt else 0
            if not is_abstain:
                selective_total += 1
                selective_correct += 1 if pred == gt else 0

    cacc = correct / total if total else 0.0
    sel_acc = selective_correct / selective_total if selective_total else 0.0
    coverage = predicted_cases / n if n else 0.0

    # macro/micro AUC
    macro_aucs = []
    for j in range(c):
        yj = [row[j] for row in y_true_mat]
        pj = [row[j] for row in prob_mat]
        macro_aucs.append(auc_binary(yj, pj))
    auc_macro = mean(macro_aucs)

    y_flat = [y for row in y_true_mat for y in row]
    p_flat = [p for row in prob_mat for p in row]
    auc_micro = auc_binary(y_flat, p_flat)

    ece = ece_multilabel(y_true_mat, prob_mat)
    brier = brier_multilabel(y_true_mat, prob_mat)

    return {
        "cACC": cacc,
        "SelectiveAcc": sel_acc,
        "Coverage": coverage,
        "AUC_macro": auc_macro,
        "AUC_micro": auc_micro,
        "ECE": ece,
        "Brier": brier,
    }


def row_to_binary(labels: List[str], label_space: List[str]) -> List[int]:
    s = set(labels)
    return [1 if lb in s else 0 for lb in label_space]


def run(args):
    rng = random.Random(args.seed)
    samples = read_samples(args.metadata_csv, args.seed)
    if not samples:
        raise ValueError("No samples loaded")

    label_space = DEFAULT_LABELS if not args.labels else [x.strip() for x in args.labels.split(",") if x.strip()]

    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]
    if not train or not test:
        raise ValueError("Need non-empty train/test splits")

    print(f"Loaded samples: total={len(samples)} train={len(train)} test={len(test)}")
    print("Extracting features...")
    feats = compute_feature_bank(samples)

    centroids = fit_centroids(train, feats, label_space)

    methods = {
        "BLIP2_zero_shot": [],
        "FineR_fixed_attr": [],
        "Med_FineR_pp": [],
    }
    abstentions = {"Med_FineR_pp": []}

    pred_rows = []
    for s in test:
        feat = feats[s.image_path]
        y_true_bin = row_to_binary(s.labels, label_space)

        base = baseline_blip2_like(s, feat, centroids, label_space)
        fixed = fixed_attribute_scores(feat, label_space)
        med, u_total = adaptive_med_finer_pp(s, feat, base, label_space, rng, args.mc_samples)
        med_abstain = u_total > args.theta

        b_probs = [base[lb] for lb in label_space]
        f_probs = [fixed[lb] for lb in label_space]
        m_probs = [med[lb] for lb in label_space]

        methods["BLIP2_zero_shot"].append((y_true_bin, b_probs))
        methods["FineR_fixed_attr"].append((y_true_bin, f_probs))
        methods["Med_FineR_pp"].append((y_true_bin, m_probs))
        abstentions["Med_FineR_pp"].append(med_abstain)

        pred_rows.append(
            {
                "image_path": s.image_path,
                "labels": "|".join(s.labels) if s.labels else "No Finding",
                "u_total": f"{u_total:.6f}",
                "abstain": "1" if med_abstain else "0",
            }
        )

    metrics = {}
    for name, rows in methods.items():
        y_true_mat = [x[0] for x in rows]
        prob_mat = [x[1] for x in rows]
        if name in abstentions:
            met = compute_metrics(y_true_mat, prob_mat, abstain=abstentions[name], threshold=args.prob_threshold)
        else:
            met = compute_metrics(y_true_mat, prob_mat, abstain=None, threshold=args.prob_threshold)
        metrics[name] = met

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-csv", required=True, help="CSV with image_path,labels[,split]")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--labels", default="", help="comma separated label set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mc-samples", type=int, default=12)
    p.add_argument("--theta", type=float, default=1.15, help="abstention threshold")
    p.add_argument("--prob-threshold", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

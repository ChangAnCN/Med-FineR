#!/usr/bin/env python3
"""Prepare binary pneumonia split CSVs from CXR8 metadata.

Expected inputs:
- Data_Entry_2017_v2020.csv (or Data_Entry_2017.csv)
- train_val_list.txt
- test_list.txt
- image root directory

Outputs:
- <out-dir>/train.csv
- <out-dir>/val.csv
- <out-dir>/test.csv
with columns: image_path,label
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_lines(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def pneumonia_label(label_text: str) -> int:
    labels = {x.strip() for x in str(label_text).split("|")}
    return int("Pneumonia" in labels)


def build_image_index(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for cur, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                index.setdefault(fn, str(Path(cur) / fn))
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-entry-csv", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--test-list", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", default="data/cxr8_pneumonia")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data_entry_csv)
    required = {"Image Index", "Finding Labels"}
    if not required.issubset(df.columns):
        raise ValueError("data-entry CSV must include 'Image Index' and 'Finding Labels'")

    train_names = read_lines(args.train_list)
    test_names = read_lines(args.test_list)

    df = df[["Image Index", "Finding Labels"]].copy()
    df["label"] = df["Finding Labels"].map(pneumonia_label)

    train_df = df[df["Image Index"].isin(train_names)].copy()
    test_df = df[df["Image Index"].isin(test_names)].copy()

    tr_df, va_df = train_test_split(
        train_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=train_df["label"],
    )

    root = Path(args.image_root)
    image_index = build_image_index(root)
    if not image_index:
        raise RuntimeError(f"No image files found under {root}")

    for part in (tr_df, va_df, test_df):
        missing = [n for n in part["Image Index"].tolist() if n not in image_index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing, first 5: {missing[:5]}")
        part["image_path"] = part["Image Index"].map(image_index.get)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr_df[["image_path", "label"]].to_csv(out / "train.csv", index=False)
    va_df[["image_path", "label"]].to_csv(out / "val.csv", index=False)
    test_df[["image_path", "label"]].to_csv(out / "test.csv", index=False)

    print(f"train={len(tr_df)} pos_rate={tr_df['label'].mean():.4f}")
    print(f"val={len(va_df)} pos_rate={va_df['label'].mean():.4f}")
    print(f"test={len(test_df)} pos_rate={test_df['label'].mean():.4f}")
    print(f"output_dir={out}")


if __name__ == "__main__":
    main()

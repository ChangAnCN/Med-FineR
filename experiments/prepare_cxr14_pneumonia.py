#!/usr/bin/env python3
"""Prepare binary pneumonia split CSVs from NIH ChestX-ray14 metadata.

Expected inputs:
- Data_Entry_2017_v2020.csv (or Data_Entry_2017.csv)
- train_val_list.txt
- test_list.txt
- images root folder containing image files

Outputs:
- out/train.csv
- out/val.csv
- out/test.csv
with columns: image_path,label
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_lines(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {x.strip() for x in f if x.strip()}


def find_image_path(image_root: Path, image_name: str) -> str:
    candidates = list(image_root.rglob(image_name))
    if not candidates:
        raise FileNotFoundError(f"Cannot find image {image_name} under {image_root}")
    return str(candidates[0])


def to_binary(label_str: str) -> int:
    labels = {x.strip() for x in str(label_str).split("|")}
    return int("Pneumonia" in labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-entry-csv", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--test-list", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", default="data/cxr14_pneumonia")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data_entry_csv)
    if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
        raise ValueError("CSV missing required columns: 'Image Index', 'Finding Labels'")

    train_set = load_lines(args.train_list)
    test_set = load_lines(args.test_list)

    df = df[["Image Index", "Finding Labels"]].copy()
    df["label"] = df["Finding Labels"].map(to_binary)

    train_df = df[df["Image Index"].isin(train_set)].copy()
    test_df = df[df["Image Index"].isin(test_set)].copy()

    tr_df, va_df = train_test_split(
        train_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=train_df["label"],
    )

    image_root = Path(args.image_root)
    for split_df in (tr_df, va_df, test_df):
        split_df["image_path"] = split_df["Image Index"].map(lambda x: find_image_path(image_root, x))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_df[["image_path", "label"]].to_csv(out_dir / "train.csv", index=False)
    va_df[["image_path", "label"]].to_csv(out_dir / "val.csv", index=False)
    test_df[["image_path", "label"]].to_csv(out_dir / "test.csv", index=False)

    print(f"Wrote {out_dir / 'train.csv'} rows={len(tr_df)} pos={tr_df['label'].mean():.4f}")
    print(f"Wrote {out_dir / 'val.csv'} rows={len(va_df)} pos={va_df['label'].mean():.4f}")
    print(f"Wrote {out_dir / 'test.csv'} rows={len(test_df)} pos={test_df['label'].mean():.4f}")


if __name__ == "__main__":
    main()

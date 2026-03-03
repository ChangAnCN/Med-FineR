#!/usr/bin/env python3
"""Fill selected LaTeX table rows in paper/v7.tex from results/metrics.json.

This updater is conservative: it only rewrites rows that start with known method
prefixes in specific tables, preserving the rest of the manuscript.
"""

from __future__ import annotations

import argparse
import json
import re


def fmt(x: float) -> str:
    return f"{100.0 * x:.2f}"


def replace_row(block: str, pattern: str, replacement: str) -> str:
    if not re.search(pattern, block, flags=re.MULTILINE):
        return block
    return re.sub(pattern, lambda _m: replacement, block, flags=re.MULTILINE)


def replace_rows_in_labeled_table(text: str, label: str, replacements: list[tuple[str, str]]) -> str:
    table_re = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.DOTALL)
    for m in table_re.finditer(text):
        block = m.group(0)
        if f"\\label{{{label}}}" not in block:
            continue
        for pattern, replacement in replacements:
            block = replace_row(block, pattern, replacement)
        return text[: m.start()] + block + text[m.end() :]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--tex", default="paper/v7.tex")
    args = ap.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        m = json.load(f)

    blip = m["BLIP2_zero_shot"]
    med = m["Med_FineR_pp"]
    finer = m["FineR_fixed_attr"]

    with open(args.tex, "r", encoding="utf-8") as f:
        tex = f.read()

    tex = replace_rows_in_labeled_table(
        tex,
        "tab:ablation",
        [
            (
                r"^Full Med-FineR\+\+\s*&.*$",
                f"Full Med-FineR++ & {fmt(med['cACC'])} & {fmt(med['SelectiveAcc'])} & {fmt(med['Coverage'])} \\\\",
            ),
            (
                r"^Fixed attributes\s*&.*$",
                f"Fixed attributes & {fmt(finer['cACC'])} & {fmt(finer['SelectiveAcc'])} & {fmt(finer['Coverage'])} \\\\",
            ),
        ],
    )

    tex = replace_rows_in_labeled_table(
        tex,
        "tab:calib",
        [
            (
                r"^BLIP-2 zero-shot\s*&.*$",
                f"BLIP-2 zero-shot & {fmt(blip['ECE'])} & {fmt(blip['Brier'])} \\\\",
            ),
            (
                r"^Med-FineR\+\+ \(ours\)\s*&.*$",
                f"Med-FineR++ (ours) & {fmt(med['ECE'])} & {fmt(med['Brier'])} \\\\",
            ),
        ],
    )

    tex = replace_rows_in_labeled_table(
        tex,
        "tab:proj",
        [
            (
                r"^BLIP-2 zero-shot\s*&.*$",
                f"BLIP-2 zero-shot & {fmt(blip['cACC'])} & -- \\\\",
            ),
            (
                r"^FineR~\\cite\{finer2024\}\s*&.*$",
                f"FineR~\\cite{{finer2024}} & {fmt(finer['cACC'])} & -- \\\\",
            ),
            (
                r"^Med-FineR\+\+ \(ours\)\s*&.*$",
                f"Med-FineR++ (ours) & {fmt(med['cACC'])} & {fmt(med['SelectiveAcc'])}@{fmt(med['Coverage'])} \\\\",
            ),
        ],
    )

    with open(args.tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Updated {args.tex} from {args.metrics}")


if __name__ == "__main__":
    main()

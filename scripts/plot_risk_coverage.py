#!/usr/bin/env python3
from __future__ import annotations
import json
import argparse
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--in-json', required=True)
    ap.add_argument('--out-png', required=True)
    args=ap.parse_args()

    with open(args.in_json,'r',encoding='utf-8') as f:
        rows=json.load(f)

    cov=[r['coverage'] for r in rows]
    acc=[r['selective_acc'] for r in rows]
    risk=[1.0-a for a in acc]

    plt.figure(figsize=(6.2,3.4), dpi=180)
    ax1=plt.gca()
    ax1.plot(cov, acc, color='#1f77b4', linewidth=2.0, label='Selective Accuracy')
    ax1.set_xlabel('Coverage')
    ax1.set_ylabel('Selective Accuracy', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xlim(0.45,1.0)
    ax1.set_ylim(0.5,0.9)
    ax1.grid(alpha=0.25, linestyle='--')

    ax2=ax1.twinx()
    ax2.plot(cov, risk, color='#d62728', linewidth=1.8, linestyle='-.', label='Risk (1-Acc)')
    ax2.set_ylabel('Risk', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0.1,0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(args.out_png, bbox_inches='tight')


if __name__=='__main__':
    main()

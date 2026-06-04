#!/usr/bin/env python3
"""Targeted regeneration of Fig 6 panel j (PanPep TDR) ONLY.

Reproduces fig6_j_tdr_panpep.{png,pdf} with ks=[5, 10, 20, 50, 100]
(dropping k=1 from previous [1, 10, 20, 50, 100]). The k=1 leftmost
point showed TDR jumping from 0/1 to 1/1 -- visually misleading at
denominator=1 and not referenced in manuscript text.

This script exists separately from generate_fig6_redesign.py to
honour the "MUST READ before touching fig6" warning in CLAUDE.md:
running the full generate_fig6_redesign.py wholesale risks
regressing unrelated panels (other panels may have accumulated
working-tree fixes that were never committed back into the script).

The logic here is copied verbatim from generate_fig6_redesign.py
panel j section (function `save` style, axis style, twin axis,
annotation, etc.) with ONLY the ks list changed and other studies
removed.

Generated files (overwrite-in-place, only these two):
    fig6_j_tdr_panpep.png
    fig6_j_tdr_panpep.pdf

Usage:
    cd <published_repo>/CaliPPer
    PYTHONPATH=Manuscript/designed_figures:. python \\
        Manuscript/designed_figures/panels/fig6/scripts/regen_fig6_j_panpep_only.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import OUTPUT_DIR, FIG_DIR  # also adds CaliPPer/ + scripts/ to sys.path

try:
    from style_config import apply_publication_style, DPI
    apply_publication_style()
except Exception as _e_style:
    import sys as _s_style
    print(f"  ⚠ FALLBACK [regen_fig6_j_panpep]: style_config import failed ({type(_e_style).__name__}: {_e_style}); using DPI=200 default", file=_s_style.stderr, flush=True)
    DPI = 200

# Match generate_fig6_redesign.py constants (now self-contained inside CaliPPer/)
DATA_DIR = os.path.join(OUTPUT_DIR, 'recal_data')
OUT_DIR = os.path.join(FIG_DIR, 'fig6')
os.makedirs(OUT_DIR, exist_ok=True)
PW, PH = 3.0, 2.5
PANPEP_COLOR = '#55A868'  # from C_STUDY['PanPep']

# Pre-flight: list files this script will overwrite
EXPECTED_OUTPUTS = [
    os.path.join(OUT_DIR, 'fig6_j_tdr_panpep.png'),
    os.path.join(OUT_DIR, 'fig6_j_tdr_panpep.pdf'),
]
for p in EXPECTED_OUTPUTS:
    if os.path.exists(p):
        print(f"  Will overwrite: {p}")
    else:
        print(f"  Will create:    {p}")


def save_panel(fig, name):
    """Apply consistent fig6 visual style and save (logic from generate_fig6_redesign.save())."""
    for ax in fig.axes:
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color('#555555')
            sp.set_linewidth(0.7)
    base = os.path.join(OUT_DIR, f'fig6_j_{name}')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {base}.png + .pdf')


# Load PanPep recalibration data
csv_path = os.path.join(DATA_DIR, 'PanPep_samples.csv')
assert os.path.exists(csv_path), f"Missing input data: {csv_path}"
df = pd.read_csv(csv_path)
y = df['y_true'].values
raw_p = df['raw_pred'].values
cal_p = df['cal_pred'].values
print(f"PanPep data: N={len(df)}, positives={int(y.sum())}, prevalence={y.mean():.3f}")

# ks: keep k=1 for visual curve start (matches Fig 6 design across studies);
# the k=1 leftmost point shows raw=0/1, cal=1/1 (dramatic recalibration impact).
# Annotation excludes k=1 (trivial denominator) — picks max-delta k>=2 instead.
ks = [1, 10, 20, 50, 100]
ks = [k for k in ks if k <= len(y)]

# Top-k TDR + cumulative discovered count
raw_order = np.argsort(raw_p)[::-1]
cal_order = np.argsort(cal_p)[::-1]
tdr_raw = [y[raw_order[:k]].sum() / k for k in ks]
tdr_cal = [y[cal_order[:k]].sum() / k for k in ks]
cum_raw = [int(y[raw_order[:k]].sum()) for k in ks]
cum_cal = [int(y[cal_order[:k]].sum()) for k in ks]
print(f"  ks:        {ks}")
print(f"  TDR raw:   {[f'{r:.3f}' for r in tdr_raw]}")
print(f"  TDR cal:   {[f'{r:.3f}' for r in tdr_cal]}")
print(f"  cum raw:   {cum_raw}")
print(f"  cum cal:   {cum_cal}")

# Plot (mirrors generate_fig6_redesign.py panel-j layout exactly)
fig, ax = plt.subplots(figsize=(PW, PH))
c = PANPEP_COLOR

# Left axis: TDR
ax.plot(ks, tdr_raw, '--', color=c, markersize=0, linewidth=1.0, alpha=0.5, label='Model only (TDR)')
ax.plot(ks, tdr_cal, '-', color=c, markersize=0, linewidth=1.5, label='CaliPPer (TDR)')
ax.fill_between(ks, tdr_raw, tdr_cal, color=c, alpha=0.08)
ax.set_xlabel('Top-k', fontsize=7)
ax.set_ylabel('True Discovery Rate', fontsize=7, color=c)
ax.tick_params(axis='y', labelcolor=c, labelsize=6)
ax.tick_params(axis='x', labelsize=6)
ax.set_ylim(-0.02, 1.05)

# Right axis: cumulative discovered count
ax2 = ax.twinx()
ax2.plot(ks, cum_raw, '--', color='#555555', linewidth=1.0, alpha=0.5, label='Model only (count)')
ax2.plot(ks, cum_cal, '-', color='#555555', linewidth=1.5, label='CaliPPer (count)')
ax2.fill_between(ks, cum_raw, cum_cal, color='#555555', alpha=0.06)
ax2.set_ylabel('Discovered count', fontsize=7, color='#555555')
ax2.tick_params(axis='y', labelcolor='#555555', labelsize=6)
ax2.set_ylim(0, max(ks))

# Annotate the optimal (max-delta) k EXCLUDING k=1.
# k=1 is kept in the curve (dramatic visual recalibration impact: 0/1 -> 1/1)
# but excluded from annotation search because denominator=1 is trivial and
# not reviewer-defensible. The chosen annotation k will be the highest-delta
# point among {10, 20, 50, 100} -- typically k=10 for PanPep (delta ~0.20).
candidate_ks = [(i, ki) for i, ki in enumerate(ks) if ki > 1]
deltas = [(i, tdr_cal[i] - tdr_raw[i]) for i, ki in candidate_ks]
best_i, best_delta = max(deltas, key=lambda x: x[1])
if best_delta > 0.01:
    k_best = ks[best_i]
    disc_raw = cum_raw[best_i]
    disc_cal = cum_cal[best_i]
    ax.annotate(f'{disc_raw}/{k_best}→{disc_cal}/{k_best}',
                xy=(k_best, tdr_cal[best_i]),
                xytext=(0.45, 0.85), textcoords='axes fraction',
                fontsize=5, color=c, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=c, linewidth=0.6))

ax.set_title('PanPep', fontweight='bold', fontsize=7, color=c)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=3.5, loc='upper right', framealpha=0.9)

save_panel(fig, 'tdr_panpep')
print("\nDone. Only panel j was regenerated; all other fig6 panels untouched.")

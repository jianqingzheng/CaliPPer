#!/usr/bin/env python3
"""Generate Fig 3 panels j (TCR) and k (TCR+BCR): distance-split-subset
prediction-error comparison and per-distance-bin |error|.

RECONSTRUCTED to match the committed source-of-truth panel images
(commit b3e07dd7, 2026-05-02). The committed .py at b3e07dd7 was an
*outdated* dataset-level boxplot generator that did NOT produce the
committed PNGs — a script/output divergence (the strip+line version that
made the PNGs was never committed and was overwritten by the older
boxplot version). Per the "panels are source of truth" rule, this script
reproduces the committed panel images, not the stale committed code.

Panel j  (fig3_method_comparison_tcr.png) — "TCR CV: distance-split subsets":
  Prediction |error| per distance-split subset for the three methods
  (PAPE, M-CBPE, S2DD/CaliPPer), grouped by metric (AUROC, AP). Jittered
  strip plot; faint grey lines pair the same subset across methods;
  horizontal bar marks each method's median. Samples are grouped into
  bins by S2DD distance rather than by epitope; TCR cross-validation.

Panel k  (fig3_method_comparison_bcr.png — legacy filename) —
  "CV prediction error vs distance":
  Mean |prediction error| (AP) per distance bin (near -> far from
  training) for the three methods, TCR (solid) and BCR (dashed).

Data source: pre-computed, committed audit CSVs (no recomputation):
  - TCR: fig4/audit_baseline_comparison_128_blosum-sqrt_results.csv
         (BLOSUM-sqrt; verified to reproduce the committed b3e07dd7 panel
          j medians; manuscript rule: TCR = BLOSUM-sqrt S2DD)
  - BCR: fig4/audit_bcr_baseline_results.csv  (Levenshtein-log)
Only strategy=='distance', split=='CV' rows are used; panel j shows
AUROC + AP (the two metrics in the committed source-of-truth panel).

Output filenames (fig3_method_comparison_{tcr,bcr}.png) are kept so
assemble_figures.py fig3() (j -> _tcr, k -> _bcr) needs no rewiring.

Usage:
    cd <published_repo>/CaliPPer
    PYTHONPATH=Manuscript/designed_figures:. python Manuscript/designed_figures/panels/fig3/scripts/generate_method_comparison_boxplot.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path

try:
    from style_config import apply_publication_style, DPI
    apply_publication_style()
except ImportError:
    DPI = 200

# Audit CSVs staged inside CaliPPer at INPUT_DIR/results/fig4_audit/
FIG4_AUDIT_DIR = os.path.join(INPUT_DIR, 'results', 'fig4_audit')
OUT_DIR = os.path.join(FIG_DIR, 'fig3', 'blosum-sqrt')
os.makedirs(OUT_DIR, exist_ok=True)

TCR_CSV = os.path.join(FIG4_AUDIT_DIR, 'audit_baseline_comparison_128_blosum-sqrt_results.csv')
BCR_CSV = os.path.join(FIG4_AUDIT_DIR, 'audit_bcr_baseline_results.csv')

METHOD_COLORS = {'pape': '#55A868', 'mcbpe': '#C44E52', 's2dd': '#4C72B0'}
METHOD_LABELS = {'pape': 'PAPE', 'mcbpe': 'M-CBPE', 's2dd': 'CaliPPer'}
METHODS = ['pape', 'mcbpe', 's2dd']            # plotting order (matches committed panel)
METRIC_LABELS = {'aucroc': 'AUROC', 'ap': 'AP'}
J_METRICS = ['aucroc', 'ap']                   # committed panel j shows AUROC + AP only


def load_distance_cv(path):
    d = pd.read_csv(path)
    d = d[(d['strategy'] == 'distance') & (d['split'] == 'CV')].copy()
    d['bin'] = d['subset'].str.extract(r'(\d+)').astype(int)
    return d


def panel_j_strip(tcr, out_name):
    """Jittered strip of per-subset |error|, grouped by metric, 3 methods,
    paired connecting lines + median bar (matches committed b3e07dd7)."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.8))
    width = 0.24
    tick_pos, tick_lab = [], []
    for mi, metric in enumerate(J_METRICS):
        sub = tcr[tcr['metric'] == metric]
        center = mi * (len(METHODS) * width + 0.35)
        tick_pos.append(center + width)
        tick_lab.append(METRIC_LABELS[metric])
        err = {m: np.abs(sub[m].values - sub['actual'].values) for m in METHODS}
        xs = {}
        for ji, m in enumerate(METHODS):
            pos = center + ji * width
            jit = rng.uniform(-width * 0.32, width * 0.32, size=len(err[m]))
            xs[m] = pos + jit
        # faint paired lines (same subset across the 3 methods)
        n = len(sub)
        for i in range(n):
            ys = [err[m][i] for m in METHODS]
            if all(np.isfinite(ys)):
                ax.plot([xs[m][i] for m in METHODS], ys, '-',
                        color='#bbbbbb', linewidth=0.25, alpha=0.35, zorder=1)
        for ji, m in enumerate(METHODS):
            pos = center + ji * width
            e = err[m]
            ok = np.isfinite(e)
            ax.scatter(xs[m][ok], e[ok], s=6, c=METHOD_COLORS[m],
                       alpha=0.55, edgecolors='none', zorder=2)
            med = np.median(e[ok])
            ax.plot([pos - width * 0.4, pos + width * 0.4], [med, med],
                    color=METHOD_COLORS[m], linewidth=2.4, zorder=3)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=7)
    ax.set_ylabel('Prediction |error|', fontsize=7)
    ax.set_title('TCR CV: distance-split subsets', fontweight='bold', fontsize=12)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)  # horizontal grid only, behind data
    handles = [Line2D([0], [0], marker='o', linestyle='', color=METHOD_COLORS[m],
                      markersize=4, label=METHOD_LABELS[m]) for m in METHODS]
    ax.legend(handles=handles, fontsize=5, loc='upper right',
              framealpha=0.9, edgecolor='#ccc')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ['.png', '.pdf']:
        fig.savefig(os.path.join(OUT_DIR, out_name + ext),
                    dpi=DPI if ext == '.png' else 300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_name}")


def panel_k_perbin(tcr, bcr, out_name):
    """Mean |error| (AP) per distance bin; TCR solid, BCR dashed."""
    def per_bin(df):
        g = df[df['metric'] == 'ap']
        return {int(b): {m: np.nanmean(np.abs(x[m].values - x['actual'].values))
                         for m in METHODS}
                for b, x in g.groupby('bin')}
    t, b = per_bin(tcr), per_bin(bcr)
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.8))
    for m in METHODS:
        xt = sorted(t)
        ax.plot(xt, [t[i][m] for i in xt], '-', color=METHOD_COLORS[m],
                marker='o', markersize=3, linewidth=1.1)
        xb = sorted(b)
        ax.plot(xb, [b[i][m] for i in xb], '--', color=METHOD_COLORS[m],
                marker='s', markersize=3, linewidth=1.1)
    ax.set_xlabel('Distance bin (near → far)', fontsize=7)
    ax.set_ylabel('Mean |prediction error| (AP)', fontsize=7)
    ax.set_title('CV prediction error vs distance', fontweight='bold', fontsize=12)
    ax.tick_params(axis='both', labelsize=6)
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)  # horizontal grid only, behind data
    handles = []
    for m in METHODS:
        handles.append(Line2D([0], [0], color=METHOD_COLORS[m], marker='o',
                              markersize=3, linestyle='-',
                              label=f'{METHOD_LABELS[m]} (TCR)'))
    for m in METHODS:
        handles.append(Line2D([0], [0], color=METHOD_COLORS[m], marker='s',
                              markersize=3, linestyle='--',
                              label=f'{METHOD_LABELS[m]} (BCR)'))
    ax.legend(handles=handles, fontsize=4.5, loc='upper left', ncol=2,
              framealpha=0.9, edgecolor='#ccc')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ['.png', '.pdf']:
        fig.savefig(os.path.join(OUT_DIR, out_name + ext),
                    dpi=DPI if ext == '.png' else 300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_name}")


# === Build panels from committed audit data ===
tcr = load_distance_cv(TCR_CSV)
bcr = load_distance_cv(BCR_CSV)
print(f"TCR CV distance-split rows: {len(tcr)} (metrics {sorted(tcr['metric'].unique())})")
print(f"BCR CV distance-split rows: {len(bcr)} (metrics {sorted(bcr['metric'].unique())})")
print("Panel j medians |error|:")
for metric in J_METRICS:
    s = tcr[tcr['metric'] == metric]
    meds = {m: np.median(np.abs(s[m].values - s['actual'].values)) for m in METHODS}
    print(f"  {metric}: " + ", ".join(f"{m}={meds[m]:.3f}" for m in METHODS))

panel_j_strip(tcr, 'fig3_method_comparison_tcr')
panel_k_perbin(tcr, bcr, 'fig3_method_comparison_bcr')
print("\nDone.")

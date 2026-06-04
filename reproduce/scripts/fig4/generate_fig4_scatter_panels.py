#!/usr/bin/env python3
"""Fig 4 Panel 1: 40 scatter plots of subset-level predicted vs actual.

Layout: 5 models × 2 strategies × 2 metrics × 2 vis_types = 40 panels.
Each panel overlays CV (halfsplit) + CT (LOO, v3+v4 cal) results.

Vis types:
  - source: colored by source (fold/test set), external data (McPAS, IEDB)
            shown with distinct markers (star, diamond)
  - property: size = subset sample count, color = prevalence (coolwarm)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import (apply_publication_style, MODEL_DISPLAY, MODEL_COLORS,
                           FOLD_COLORS, FONT_LABEL, FONT_TICK, FONT_LEGEND, DPI)
sys.path.insert(0, os.path.join(INPUT_DIR, 'Manuscript', 'designed_figures', 'panels'))
from dist_config import DIST_TYPE, DIST_SUBDIR
apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig3_fig4_tcr_cache')
_FIG_DIR = os.path.join(FIG_DIR, 'fig4')
OUT_DIR = os.path.join(_FIG_DIR, DIST_SUBDIR[DIST_TYPE], 'TCR_panels')
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
STRATEGIES = ['epitope', 'distance']
METRICS = ['aucroc', 'ap']
METRIC_DISP = {'aucroc': 'AUROC', 'ap': 'AP'}
STRATEGY_DISP = {'epitope': 'Epitope splitting', 'distance': 'Distance splitting'}

# Source styles for CT test sets
CT_COLORS = {
    'seen_test': '#1f77b4', 'unseen_fold34': '#ff7f0e',
    'v3_combined': '#2ca02c', 'v4_combined': '#d62728',
    'mcpas': '#9467bd', 'iedb_sars': '#8c564b',
}
CT_DISPLAY = {
    'seen_test': 'Seen', 'unseen_fold34': 'Unseen', 'v3_combined': 'v3',
    'v4_combined': 'v4', 'mcpas': 'McPAS', 'iedb_sars': 'IEDB',
}
# External test sets get distinct markers
EXTERNAL_SETS = {'mcpas', 'iedb_sars'}
CT_MARKERS = {s: ('*' if s in EXTERNAL_SETS else 'o') for s in CT_COLORS}
CT_MARKER_SIZE = {s: (60 if s in EXTERNAL_SETS else 20) for s in CT_COLORS}

CV_COLORS = {i: FOLD_COLORS[i] for i in range(5)}
CV_DISPLAY = {i: f'F{i}' for i in range(5)}

PW, PH = 3.0, 3.0  # square for predicted vs actual


def load_cached(split, strategy, metric):
    f = os.path.join(TCR_CACHE, f'tcr_fig4_{DIST_TYPE}_{split}_{strategy}_{metric}.csv')
    if os.path.exists(f):
        return pd.read_csv(f)
    return None


def plot_source_scatter(ax, ct_df, cv_df, model, metric):
    """Scatter colored by source, shape for external."""
    all_x, all_y = [], []

    # CV points — blue tones, hollow squares
    if cv_df is not None:
        for fold in sorted(cv_df['fold'].unique()):
            sub = cv_df[cv_df['fold'] == fold]
            ax.scatter(sub['actual'], sub['predicted'],
                       facecolors='none', edgecolors=CV_COLORS.get(fold, '#aaa'),
                       marker='s', s=22, linewidths=0.8,
                       alpha=0.7, label=f'CV F{fold}', zorder=2)
            all_x.extend(sub['actual']); all_y.extend(sub['predicted'])

    # CT points — filled, circles for internal, stars for external
    if ct_df is not None:
        for src in sorted(ct_df['source'].unique()):
            sub = ct_df[ct_df['source'] == src]
            ax.scatter(sub['actual'], sub['predicted'],
                       c=CT_COLORS.get(src, '#aaa'),
                       marker=CT_MARKERS.get(src, 'o'),
                       s=CT_MARKER_SIZE.get(src, 25),
                       alpha=0.8, edgecolors='white', linewidths=0.5,
                       label=f'CT {CT_DISPLAY.get(src, src)}', zorder=3)
            all_x.extend(sub['actual']); all_y.extend(sub['predicted'])

    _finish_scatter(ax, np.array(all_x), np.array(all_y), metric)
    ax.legend(fontsize=FONT_LEGEND - 3, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              framealpha=0.85, handletextpad=0.3, borderpad=0.3, labelspacing=0.2,
              borderaxespad=0)


def plot_property_scatter(ax, ct_df, cv_df, metric):
    """Scatter: size = n, color = prevalence."""
    all_x, all_y, all_prev, all_n = [], [], [], []

    for df in [cv_df, ct_df]:
        if df is None:
            continue
        all_x.extend(df['actual']); all_y.extend(df['predicted'])
        all_prev.extend(df['prevalence']); all_n.extend(df['n'])

    if not all_x:
        return

    all_x, all_y = np.array(all_x), np.array(all_y)
    all_prev, all_n = np.array(all_prev), np.array(all_n, dtype=float)

    # Size scaling
    n_min, n_max = all_n.min(), all_n.max()
    if n_max > n_min:
        sizes = 12 + 100 * (all_n - n_min) / (n_max - n_min)
    else:
        sizes = np.full_like(all_n, 30)

    sc = ax.scatter(all_x, all_y, c=all_prev, cmap='coolwarm',
                    vmin=0, vmax=1, s=sizes,
                    alpha=0.7, edgecolors='white', linewidths=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label('Class prevalence', fontsize=FONT_LEGEND - 2)
    cbar.ax.tick_params(labelsize=FONT_LEGEND - 3)

    _finish_scatter(ax, all_x, all_y, metric)

    # Size legend (3 levels)
    n_mid = (n_min + n_max) / 2
    for n_val in [n_min, n_mid, n_max]:
        s = 12 + 100 * (n_val - n_min) / (n_max - n_min + 1e-9)
        ax.scatter([], [], c='gray', s=s, alpha=0.5, label=f'n={int(n_val)}')
    ax.legend(fontsize=FONT_LEGEND - 3, loc='lower right', framealpha=0.85,
              handletextpad=0.2, borderpad=0.2, labelspacing=0.2)


def _finish_scatter(ax, all_x, all_y, metric):
    if len(all_x) < 3:
        return
    pad = 0.03
    lo = min(all_x.min(), all_y.min()) - pad
    hi = max(all_x.max(), all_y.max()) + pad
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    r, p = pearsonr(all_x, all_y)
    mae = np.mean(np.abs(all_x - all_y))
    n = len(all_x)
    p_str = f'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    ax.text(0.05, 0.95, f'r = {r:.2f}, {p_str}\nMAE = {mae:.3f}, n = {n}',
            transform=ax.transAxes, fontsize=FONT_TICK - 2, va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#ccc', alpha=0.9, linewidth=0.3))
    mdisp = METRIC_DISP.get(metric, metric)
    ax.set_xlabel(f'Actual {mdisp}', fontsize=FONT_LABEL - 2)
    ax.set_ylabel(f'Predicted {mdisp}', fontsize=FONT_LABEL - 2)


# ── Generate all 40 panels ──
print("Generating 40 scatter panels...")
count = 0
for model in MODELS:
    for strategy in STRATEGIES:
        for metric in METRICS:
            # Load data
            ct_all = load_cached('ct', strategy, metric)
            cv_all = load_cached('cv', strategy, metric)
            ct_df = ct_all[ct_all['model'] == model] if ct_all is not None and 'model' in ct_all.columns else None
            cv_df = cv_all[cv_all['model'] == model] if cv_all is not None and 'model' in cv_all.columns else None

            if (ct_df is None or len(ct_df) == 0) and (cv_df is None or len(cv_df) == 0):
                print(f"  SKIP {model} {strategy} {metric} — no data")
                continue

            mdisp = METRIC_DISP[metric]
            sdisp = STRATEGY_DISP[strategy]
            model_disp = MODEL_DISPLAY[model]

            for vis in ['source', 'property']:
                fig, ax = plt.subplots(1, 1, figsize=(PW, PH))
                if vis == 'source':
                    plot_source_scatter(ax, ct_df, cv_df, model, metric)
                else:
                    plot_property_scatter(ax, ct_df, cv_df, metric)

                ax.set_title(f'{model_disp}: {sdisp.lower()}, {mdisp}',
                             fontsize=FONT_TICK, fontweight='bold')

                out = os.path.join(OUT_DIR,
                    f'fig4_{vis}_{strategy}_{metric}_{model}')
                fig.savefig(out + '.png', dpi=DPI, bbox_inches='tight')
                fig.savefig(out + '.pdf', dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                count += 1

print(f"Done: {count} panels saved to {OUT_DIR}")

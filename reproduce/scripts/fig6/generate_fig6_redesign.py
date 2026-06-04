#!/usr/bin/env python3
"""Generate ALL fig6 panels for the redesigned retrospective validation figure.

Requires: compute_fig6_recal_data.py to be run first (creates recal_data/).

Usage:
    cd <published_repo>/CaliPPer
    PYTHONPATH=Manuscript/designed_figures:. python Manuscript/designed_figures/panels/fig6/scripts/generate_fig6_redesign.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, FIG_DIR  # also adds CaliPPer/ + scripts/ to sys.path

try:
    from style_config import apply_publication_style, DPI
    apply_publication_style()
except Exception as _e_style:
    import sys as _s_style
    print(f"  ⚠ FALLBACK [fig6_redesign]: style_config import failed ({type(_e_style).__name__}: {_e_style}); using DPI=200 default", file=_s_style.stderr, flush=True)
    DPI = 200

from calipper.general_evaluator import safe_metric

RESULTS = os.path.join(INPUT_DIR, 'results')
DATA_DIR = os.path.join(OUTPUT_DIR, 'recal_data')
OUT_DIR = os.path.join(FIG_DIR, 'fig6')
os.makedirs(OUT_DIR, exist_ok=True)
PW, PH = 3.0, 2.5

C_STUDY = {
    'deepAntigen': '#4C72B0',
    'XBCR-net': '#DD8452',
    'PanPep': '#55A868',
    'BigMHC': '#C44E52',
    'AntibioticsAI': '#8172B3',
}
STUDY_ORDER = ['XBCR-net', 'deepAntigen', 'AntibioticsAI', 'PanPep', 'BigMHC']
METRIC_MARKERS = {'aucroc': 'o', 'ap': '^', 'f1': 's'}
METHOD_COLORS = {'caliper': '#4C72B0', 'pape': '#55A868', 'mcbpe': '#C44E52'}


def save(fig, label, name):
    # Apply consistent fig6 visual style: subtle grid + visible 4-spine border
    # on every axes in the figure. Spine color #555 lw=0.7 (clearly visible);
    # grid #cccccc alpha=0.3 lw=0.4 (visible but not obtrusive).
    # Placeholders (panels a, b) and explicit-spine-off panels are excluded
    # via the `_skip_style` attribute below.
    for ax in fig.axes:
        if getattr(ax, '_skip_style', False):
            continue
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color('#555555')
            sp.set_linewidth(0.7)
    base = os.path.join(OUT_DIR, f'fig6_{label}_{name}')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{label}] {name}')


def load_study(name):
    path = os.path.join(DATA_DIR, f'{name}_samples.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


# Load summary
summary = pd.read_csv(os.path.join(DATA_DIR, 'recal_summary_all.csv'))


# ═══════════════════════════════════════════
# A, B: Placeholders
# ═══════════════════════════════════════════
for label in ['a', 'b']:
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.text(0.5, 0.5, 'Conceptual\ndiagram', ha='center', va='center',
            fontsize=14, color='#666', fontstyle='italic', transform=ax.transAxes)
    ax.set_facecolor('#f0f0f0'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#ccc')
    ax._skip_style = True  # Placeholders keep light-gray frame, no grid
    save(fig, label, 'placeholder')


# ═══════════════════════════════════════════
# C: Prediction scatter (AUROC + AP)
# ═══════════════════════════════════════════
print("\n=== Panel C: Prediction scatter (AUROC + AP) ===")
fig, ax = plt.subplots(figsize=(PW, PH))
# Prediction data: (study, metric, actual, predicted)
# Sources: per-study performance_prediction CSVs + BLOSUM-sqrt values for short-seq studies
# Read from compute_fig6_panel_c_d.py output (fixes XBCR label leakage + AntibioticsAI unfair comparison)
_pred_csv = os.path.join(DATA_DIR, 'fig6_panel_c_predictions.csv')
if os.path.exists(_pred_csv):
    _pdf = __import__('pandas').read_csv(_pred_csv)
    pred_points = [(r['study'], r['metric'], r['actual'], r['predicted']) for _, r in _pdf.iterrows()]
else:
    raise FileNotFoundError(f"Run compute_fig6_panel_c_d.py first: {_pred_csv}")
auroc_errs, ap_errs = [], []
for study, metric, actual, predicted in pred_points:
    # Map sub-study names to parent study color (e.g. deepAntigen_neo → deepAntigen)
    parent_study = study.split('_neo')[0] if '_neo' in study else study
    color = C_STUDY.get(parent_study, '#888')
    # Use hollow marker for _neo variants to distinguish from main test
    marker = METRIC_MARKERS[metric]
    alpha = 0.4 if '_neo' in study else 1.0
    ax.scatter(actual, predicted, c=color, marker=marker,
               s=40 if '_neo' not in study else 25, alpha=alpha,
               edgecolor='white', linewidth=0.3, zorder=3)
    # Only include main study (not _neo) in MAE calculation for Panel D header
    if '_neo' not in study:
        err = abs(predicted - actual)
        if metric == 'aucroc': auroc_errs.append(err)
        else: ap_errs.append(err)

ax.plot([0, 1.05], [0, 1.05], 'k--', linewidth=0.5, alpha=0.3)
ax.text(0.05, 0.90, f'AUROC MAE={np.mean(auroc_errs):.3f}\nAP MAE={np.mean(ap_errs):.3f}',
        transform=ax.transAxes, fontsize=5, va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.9, linewidth=0.3))
ax.set_xlabel('Actual', fontsize=7); ax.set_ylabel('Predicted', fontsize=7)
ax.set_title('Performance prediction\n(5 retrospective studies)', fontweight='bold', fontsize=7)
ax.tick_params(labelsize=6)
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#888', markersize=5, label='AUROC'),
           Line2D([0],[0], marker='^', color='w', markerfacecolor='#888', markersize=5, label='AP')]
for s in STUDY_ORDER:
    handles.append(Line2D([0],[0], marker='s', color='w', markerfacecolor=C_STUDY[s], markersize=4, label=s))
ax.legend(handles=handles, fontsize=3.5, loc='lower right', ncol=2)
save(fig, 'c', 'prediction_scatter')


# ═══════════════════════════════════════════
# E: Recalibration dumbbell (AUROC top, AP bottom — fig5 I/J style)
# ═══════════════════════════════════════════
print("\n=== Panel E: Recal dumbbell AUROC+AP ===")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PW + 0.3, PH + 0.8), sharex=True,
                                gridspec_kw={'hspace': 0.4})
for ax, metric, title in [(ax1, 'aucroc', 'AUROC (5 studies)'), (ax2, 'ap', 'AP (5 studies)')]:
    recal = []
    for study in STUDY_ORDER:
        row = summary[(summary['study'] == study) & (summary['metric'] == metric)]
        if len(row) > 0:
            r = row.iloc[0]
            recal.append((study, r['before'], r['after']))
    recal.sort(key=lambda x: x[2] - x[1], reverse=True)
    yp = np.arange(len(recal))[::-1]
    for i, (s, b, a) in enumerate(recal):
        c = C_STUDY[s]
        ax.plot([b, a], [yp[i], yp[i]], color=c, linewidth=2.5, solid_capstyle='round', alpha=0.5)
        ax.scatter(b, yp[i], color='white', edgecolor=c, s=30, zorder=5, linewidth=1)
        ax.scatter(a, yp[i], color=c, s=35, zorder=5, edgecolor='white', linewidth=0.4)
        ax.text(max(a, b) + 0.01, yp[i], f'{a-b:+.3f}', va='center', fontsize=6,
                color=c, fontweight='bold')
    ax.axvline(0.5, color='gray', linewidth=0.3, linestyle=':', alpha=0.4)
    ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in recal], fontsize=6)
    ax.set_title(title, fontweight='bold', fontsize=8)
    ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', labelsize=6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax2.set_xlabel('Metric value', fontsize=7)
ax1.legend(handles=[
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='#666', markersize=5, label='Before'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#666', markersize=5, label='After recalib.'),
], fontsize=5, loc='lower right', framealpha=0.9)
save(fig, 'e', 'recal_dumbbell_auroc_ap')


# F: Before/after scatter REMOVED — replaced by deepAntigen ROC in new layout


# ═══════════════════════════════════════════
# G, I, K, M, O: ROC curves (5 studies)
# ═══════════════════════════════════════════
print("\n=== ROC panels ===")
roc_panels = [('f', 'deepAntigen'), ('i', 'PanPep'), ('k', 'XBCR-net'),
              ('m', 'BigMHC'), ('o', 'AntibioticsAI')]

for label, study in roc_panels:
    df = load_study(study)
    if df is None:
        print(f"  [{label}] {study}: data not found")
        continue
    fig, ax = plt.subplots(figsize=(PW, PH))
    c = C_STUDY[study]

    y = df['y_true'].values
    if len(np.unique(y)) < 2:
        ax.text(0.5, 0.5, f'{study}\nInsufficient classes', ha='center', va='center', transform=ax.transAxes)
        save(fig, label, f'roc_{study.lower().replace("-", "")}')
        continue

    # Before
    fpr_b, tpr_b, _ = roc_curve(y, df['raw_pred'].values)
    auc_b = roc_auc_score(y, df['raw_pred'].values)
    # After
    fpr_a, tpr_a, _ = roc_curve(y, df['cal_pred'].values)
    auc_a = roc_auc_score(y, df['cal_pred'].values)

    ax.plot(fpr_b, tpr_b, '--', color=c, linewidth=1, alpha=0.6, label=f'Before ({auc_b:.3f})')
    ax.plot(fpr_a, tpr_a, '-', color=c, linewidth=1.5, label=f'After ({auc_a:.3f})')
    # Fill between curves: trace exact curve points as polygon
    # (np.interp smooths step boundaries; polygon follows exact line segments)
    poly_x = np.concatenate([fpr_a, fpr_b[::-1]])
    poly_y = np.concatenate([tpr_a, tpr_b[::-1]])
    ax.fill(poly_x, poly_y, color=c, alpha=0.1)
    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('FPR', fontsize=7); ax.set_ylabel('TPR', fontsize=7)
    ax.set_title(f'{study}\n(ΔAUROC +{auc_a-auc_b:.3f})', fontweight='bold', fontsize=7, color=c)
    ax.legend(fontsize=5, loc='lower right')
    ax.tick_params(labelsize=6)
    save(fig, label, f'roc_{study.lower().replace("-", "").replace(" ", "")}')


# ═══════════════════════════════════════════
# J: Neoantigen reranking scatter
# ═══════════════════════════════════════════
print("\n=== Panel G: Neoantigen reranking ===")
da = load_study('deepAntigen')
if da is not None:
    fig, ax = plt.subplots(figsize=(PW, PH))
    y = da['y_true'].values
    raw_rank = np.argsort(np.argsort(da['raw_pred'].values)[::-1]) + 1
    cal_rank = np.argsort(np.argsort(da['cal_pred'].values)[::-1]) + 1

    c_dark = C_STUDY['deepAntigen']
    c_light = '#a8c4e0'

    for i in range(len(y)):
        c = c_dark if y[i] == 1 else c_light
        s = 25 if y[i] == 1 else 8
        ax.scatter(raw_rank[i], cal_rank[i], c=c, s=s, edgecolor='white', linewidth=0.2, zorder=3 if y[i] else 2)

    ax.plot([0, len(y) + 1], [0, len(y) + 1], 'k--', linewidth=0.5, alpha=0.3)
    promoted = sum(1 for i in range(len(y)) if y[i] == 1 and cal_rank[i] < raw_rank[i])
    n_pos = int(y.sum())
    ax.text(0.95, 0.05, f'{promoted}/{n_pos} confirmed\npromoted',
            transform=ax.transAxes, fontsize=5, va='bottom', ha='right', color=c_dark, fontweight='bold',
            fontstyle='italic')
    ax.set_xlabel('Raw model rank', fontsize=7); ax.set_ylabel('CaliPPer rank', fontsize=7)
    ax.set_title('deepAntigen neoantigen\nreranking', fontweight='bold', fontsize=7, color=c_dark)
    ax.tick_params(labelsize=6)
    from matplotlib.lines import Line2D
    legend_el = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_dark, markersize=5, label='Confirmed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_light, markersize=4, label='Non-confirmed'),
    ]
    ax.legend(handles=legend_el, fontsize=4, loc='upper left', framealpha=0.9)
    save(fig, 'g', 'neoantigen_reranking')


# ═══════════════════════════════════════════
# H: PanPep TDR + N: BigMHC TDR + P: AntibioticsAI TDR
# ═══════════════════════════════════════════
print("\n=== TDR panels ===")
tdr_panels = [('h', 'deepAntigen'), ('j', 'PanPep'), ('n', 'BigMHC'), ('p', 'AntibioticsAI')]

for label, study in tdr_panels:
    fig, ax = plt.subplots(figsize=(PW, PH))
    c = C_STUDY[study]

    # 2026-05-21: AntibioticsAI TDR now uses the FLIPPED halfsplit
    # (cal=odd, test=even) from recal_data/AntibioticsAI_samples.csv,
    # matching Panel O and Panel E. Previously used full-dataset v2.7
    # self-calibration; switched to halfsplit per user request so all
    # AntibioticsAI panels (E, O, P) share one design (genuine cal/test
    # separation, no test-leakage in PPV/NPV estimation).
    df = load_study(study)
    if df is None: continue
    y = df['y_true'].values
    raw_p = df['raw_pred'].values
    cal_p = df['cal_pred'].values
    if study == 'deepAntigen':
        ks = [1, 5, 10, 20, 50, 100]
    else:
        ks = [1, 10, 20, 50, 100]

    ks = [k for k in ks if k <= len(y)]
    raw_order = np.argsort(raw_p)[::-1]
    cal_order = np.argsort(cal_p)[::-1]
    tdr_raw = [y[raw_order[:k]].sum() / k for k in ks]
    tdr_cal = [y[cal_order[:k]].sum() / k for k in ks]
    # Cumulative discovered count
    cum_raw = [y[raw_order[:k]].sum() for k in ks]
    cum_cal = [y[cal_order[:k]].sum() for k in ks]

    # Left axis: TDR (study color, dashed=before, solid=after, matching ROC style)
    ax.plot(ks, tdr_raw, '--', color=c, markersize=0, linewidth=1.0, alpha=0.5, label='Model only (TDR)')
    ax.plot(ks, tdr_cal, '-', color=c, markersize=0, linewidth=1.5, label='CaliPPer (TDR)')
    ax.fill_between(ks, tdr_raw, tdr_cal, color=c, alpha=0.08)
    ax.set_xlabel('Top-k', fontsize=7)
    ax.set_ylabel('True Discovery Rate', fontsize=7, color=c)
    ax.tick_params(axis='y', labelcolor=c, labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    # Adaptive TDR y-axis for deepAntigen (max TDR ~0.6), fixed for others
    if study == 'deepAntigen':
        tdr_max = max(max(tdr_raw), max(tdr_cal))
        ax.set_ylim(-0.02, tdr_max + 0.05)
    else:
        ax.set_ylim(-0.02, 1.05)

    # Right axis: Cumulative discovered count (dark gray, dashed=before, solid=after)
    ax2 = ax.twinx()
    ax2.plot(ks, cum_raw, '--', color='#555555', linewidth=1.0, alpha=0.5, label='Model only (count)')
    ax2.plot(ks, cum_cal, '-', color='#555555', linewidth=1.5, label='CaliPPer (count)')
    ax2.fill_between(ks, cum_raw, cum_cal, color='#555555', alpha=0.06)
    ax2.set_ylabel('Discovered count', fontsize=7, color='#555555')
    ax2.tick_params(axis='y', labelcolor='#555555', labelsize=6)
    ax2.set_ylim(0, max(ks))

    # Annotate largest TDR improvement (discovered_num/topk format)
    deltas = [tdr_cal[i] - tdr_raw[i] for i in range(len(ks))]
    best_i = max(range(len(deltas)), key=lambda i: deltas[i])
    if deltas[best_i] > 0.01:
        k_best = ks[best_i]
        disc_raw = int(cum_raw[best_i])
        disc_cal = int(cum_cal[best_i])
        ax.annotate(f'{disc_raw}/{k_best}→{disc_cal}/{k_best}',
                    xy=(k_best, tdr_cal[best_i]),
                    xytext=(0.45, 0.85), textcoords='axes fraction',
                    fontsize=5, color=c, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=c, linewidth=0.6))

    ax.set_title(f'{study}', fontweight='bold', fontsize=7, color=c)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=3.5, loc='upper right', framealpha=0.9)

    save(fig, label, f'tdr_{study.lower().replace("-", "")}')


# ═══════════════════════════════════════════
# L: XBCR Omicron reranking scatter (same format as Panel J)
# ═══════════════════════════════════════════
print("\n=== Panel L: XBCR Omicron reranking ===")
xbcr = load_study('XBCR-net')
if xbcr is not None:
    fig, ax = plt.subplots(figsize=(PW, PH))
    y = xbcr['y_true'].values
    raw_rank = np.argsort(np.argsort(xbcr['raw_pred'].values)[::-1]) + 1
    cal_rank = np.argsort(np.argsort(xbcr['cal_pred'].values)[::-1]) + 1

    c_dark = C_STUDY['XBCR-net']
    c_light = '#f0d0a8'

    for i in range(len(y)):
        c = c_dark if y[i] == 1 else c_light
        s = 25 if y[i] == 1 else 8
        ax.scatter(raw_rank[i], cal_rank[i], c=c, s=s, edgecolor='white', linewidth=0.2, zorder=3 if y[i] else 2)

    ax.plot([0, len(y) + 1], [0, len(y) + 1], 'k--', linewidth=0.5, alpha=0.3)
    promoted = sum(1 for i in range(len(y)) if y[i] == 1 and cal_rank[i] < raw_rank[i])
    n_pos = int(y.sum())
    ax.text(0.95, 0.05, f'{promoted}/{n_pos} binders\npromoted',
            transform=ax.transAxes, fontsize=5, va='bottom', ha='right', color=c_dark, fontweight='bold',
            fontstyle='italic')
    ax.set_xlabel('Raw model rank', fontsize=7); ax.set_ylabel('CaliPPer rank', fontsize=7)
    ax.set_title('XBCR-net Omicron\nreranking', fontweight='bold', fontsize=7, color=c_dark)
    ax.tick_params(labelsize=6)
    from matplotlib.lines import Line2D
    legend_el = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_dark, markersize=5, label='Binder'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_light, markersize=4, label='Non-binder'),
    ]
    ax.legend(handles=legend_el, fontsize=4, loc='upper left', framealpha=0.9)
    save(fig, 'l', 'xbcr_omicron_reranking')


# ═══════════════════════════════════════════
# D: Baseline boxplot — S2DD vs PAPE vs M-CBPE (no distance for baselines)
# Data: fig6/recal_data/fig6_prediction_3method.csv
# Generated by inline computation (deposited here for reproducibility)
# ═══════════════════════════════════════════
print("\n=== Panel D: Baseline boxplot (split AUROC/AP) ===")
pred_3m_path = os.path.join(DATA_DIR, 'fig6_prediction_3method.csv')
if os.path.exists(pred_3m_path):
    d_df = pd.read_csv(pred_3m_path)
    if 'S2DD' in d_df.columns:
        d_df = d_df.rename(columns={'S2DD': 'CaliPPer'})
    METHOD_COLORS_D = {'CaliPPer': '#4C72B0', 'PAPE': '#55A868', 'M-CBPE': '#C44E52'}
    methods_d = ['PAPE', 'MCBPE', 'CaliPPer']
    method_labels_d = {'PAPE': 'PAPE', 'MCBPE': 'M-CBPE', 'CaliPPer': 'CaliPPer'}
    metric_titles = {'aucroc': 'AUROC', 'ap': 'AP'}

    fig, (ax_auc, ax_ap) = plt.subplots(1, 2, figsize=(PW + 0.5, PH),
                                          gridspec_kw={'wspace': 0.35})

    for ax, metric in [(ax_auc, 'aucroc'), (ax_ap, 'ap')]:
        sub = d_df[(d_df['metric'] == metric) & (~d_df['study'].str.contains('_neo'))]
        width = 0.25
        for ji, method in enumerate(methods_d):
            pos = ji * width
            vals = sub[method].values
            mean_val = np.mean(vals)
            ax.bar(pos, mean_val, width * 0.85, color=METHOD_COLORS_D[method_labels_d[method]],
                   alpha=0.5, edgecolor='black', linewidth=0.3)
            ax.text(pos, mean_val + 0.003, f'{mean_val:.3f}', ha='center', fontsize=3.5, color='#444',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.7))
            for idx, (_, r) in enumerate(sub.iterrows()):
                jitter = np.random.RandomState(42 + idx).uniform(-0.03, 0.03)
                ax.scatter(pos + jitter, r[method], c=C_STUDY.get(r['study'], '#888'), s=12,
                           edgecolor='white', linewidth=0.2, zorder=5)
        # Connecting lines
        for _, r in sub.iterrows():
            xs = [ji * width for ji in range(len(methods_d))]
            ys = [r[m] for m in methods_d]
            ax.plot(xs, ys, color=C_STUDY.get(r['study'], '#888'), linewidth=0.4, alpha=0.4, zorder=1)

        ax.set_xticks([ji * width for ji in range(len(methods_d))])
        ax.set_xticklabels([method_labels_d[m] for m in methods_d], fontsize=5, rotation=30, ha='right')
        ax.set_title(metric_titles[metric], fontweight='bold', fontsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax_auc.set_ylabel('Prediction |error|', fontsize=7)
    fig.suptitle('CaliPPer vs baselines (5 retrospective studies)', fontweight='bold', fontsize=7, y=1.02)

    from matplotlib.patches import Patch
    ax_ap.legend(handles=[Patch(facecolor=METHOD_COLORS_D[method_labels_d[m]], alpha=0.5, label=method_labels_d[m])
                           for m in methods_d], fontsize=3.5, loc='upper right')
    save(fig, 'd', 'baseline_boxplot')
else:
    print("  WARNING: fig6_prediction_3method.csv not found. Run inline computation first.")


print(f"\n{'=' * 60}")
print("All panels generated.")
print(f"Output: {OUT_DIR}/fig6_{{label}}_{{name}}.png")

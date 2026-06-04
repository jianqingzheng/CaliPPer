#!/usr/bin/env python3
"""Fig6 PRP-TCR (6th retrospective) panels — reranking scatter, ROC
comparison, TDR improvement. SAME format/style as the 5-study fig6
panels (generate_fig6_redesign.py); only the colour differs (PRP-TCR =
brown #937860, vs the 5 study colours).

Reads panels/fig6/recal_data/PRP-TCR_{samples,recal_per_tcr,summary}.csv
produced by compute_fig6_prp_tcr_recal.py (v2.7 API @ the single
PRP-TCR optimum; reproduces investigation.md §24-PRP3 EXACTLY:
per-TCR mean ΔAUROC +0.0198, ΔAP +0.0188, ΣΔTDR@5 +4).

⚠ HONEST LEVEL — per-TCR, NOT pooled. CD69 is a per-patient-TCR
discovery task: each per-TCR model has its OWN score scale, so a single
pooled ROC across the 6 TCRs is a Simpson-type artifact (pooled
ΔAUROC=−0.063, NOT the result). Unlike the 5 single-test-set studies,
the meaningful level here is per-TCR (mean ΔAUROC/ΔAP) and Σ-over-TCR
TDR. The ROC panel therefore shows the 6 per-TCR curve pairs + their
vertical mean (the recorded +0.0198), never one pooled curve.

Usage:
    cd <published_repo>/CaliPPer
    PYTHONPATH=Manuscript/designed_figures:. python \
      Manuscript/designed_figures/panels/fig6/scripts/generate_fig6_prp_tcr_panels.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Manuscript', 'designed_figures'))
try:
    from style_config import apply_publication_style, DPI
    apply_publication_style()
except Exception as _e_style:
    import sys as _s_style
    print(f"  ⚠ FALLBACK [fig6_prp_tcr_panels]: style_config import failed ({type(_e_style).__name__}: {_e_style}); using DPI=200 default", file=_s_style.stderr, flush=True)
    DPI = 200

DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'recal_data')
OUT_DIR = os.path.join(SCRIPT_DIR, '..')
PW, PH = 3.0, 2.5
C_PRP = '#937860'          # PRP-TCR — the distinct 6th colour
C_PRP_LIGHT = '#cbb9a8'    # non-activator points (matches G/L light style)


def save(fig, label, name):
    base = os.path.join(OUT_DIR, f'fig6_{label}_{name}')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{label}] {name}')


s = pd.read_csv(os.path.join(DATA_DIR, 'PRP-TCR_samples.csv'))
pt = pd.read_csv(os.path.join(DATA_DIR, 'PRP-TCR_recal_per_tcr.csv'))
sm = pd.read_csv(os.path.join(DATA_DIR, 'PRP-TCR_summary.csv'))


def _g(metric, level, col):
    r = sm[(sm['metric'] == metric) & (sm['level'] == level)]
    return float(r.iloc[0][col]) if len(r) else np.nan


MEAN_DAUC = _g('aucroc', 'per_tcr_mean', 'delta')
MEAN_DAP = _g('ap', 'per_tcr_mean', 'delta')
MEAN_AUC_B = _g('aucroc', 'per_tcr_mean', 'before')
MEAN_AUC_A = _g('aucroc', 'per_tcr_mean', 'after')
N_TCR = int(pt.shape[0])


# ═══════════════════════════════════════════
# Panel 1: reranking scatter (Panel-G/L format)
# ═══════════════════════════════════════════
print("\n=== PRP-TCR reranking scatter ===")
fig, ax = plt.subplots(figsize=(PW, PH))
y = s['y_true'].values
rr = s['raw_rank_in_tcr'].values
cr = s['cal_rank_in_tcr'].values
for i in range(len(y)):
    c = C_PRP if y[i] == 1 else C_PRP_LIGHT
    sz = 25 if y[i] == 1 else 8
    ax.scatter(rr[i], cr[i], c=c, s=sz, edgecolor='white', linewidth=0.2,
               zorder=3 if y[i] else 2)
mx = int(max(rr.max(), cr.max())) + 1
ax.plot([0, mx], [0, mx], 'k--', linewidth=0.5, alpha=0.3)
promoted = int(np.sum((y == 1) & (cr < rr)))
n_act = int((y == 1).sum())
ax.text(0.95, 0.05, f'{promoted}/{n_act} activators\npromoted',
        transform=ax.transAxes, fontsize=5, va='bottom', ha='right',
        color=C_PRP, fontweight='bold', fontstyle='italic')
ax.set_xlabel('Raw model rank (within TCR)', fontsize=7)
ax.set_ylabel('CaliPPer rank (within TCR)', fontsize=7)
ax.set_title('PRP-TCR CD69\nreranking', fontweight='bold', fontsize=7,
             color=C_PRP)
ax.tick_params(labelsize=6)
ax.legend(handles=[
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_PRP,
           markersize=5, label='Activator'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_PRP_LIGHT,
           markersize=4, label='Non-activator')],
    fontsize=4, loc='upper left', framealpha=0.9)
save(fig, 'prp', 'reranking')


# ═══════════════════════════════════════════
# Panel 2: ROC comparison — PER-TCR (6 pairs + vertical mean)
# (Panel-F/I/K visual style; honest level, not pooled)
# ═══════════════════════════════════════════
print("\n=== PRP-TCR ROC comparison (per-TCR) ===")
fig, ax = plt.subplots(figsize=(PW, PH))
grid = np.linspace(0, 1, 200)
tpr_b_all, tpr_a_all = [], []
for t, g in s.groupby('tcr_id'):
    yt = g['y_true'].values
    if len(np.unique(yt)) < 2:
        continue
    fb, tb, _ = roc_curve(yt, g['raw_pred'].values)
    fa, ta, _ = roc_curve(yt, g['cal_pred'].values)
    ax.plot(fb, tb, '--', color=C_PRP, linewidth=0.5, alpha=0.20,
            zorder=1)
    ax.plot(fa, ta, '-', color=C_PRP, linewidth=0.5, alpha=0.20,
            zorder=1)
    tpr_b_all.append(np.interp(grid, fb, tb))
    tpr_a_all.append(np.interp(grid, fa, ta))
mean_b = np.mean(tpr_b_all, axis=0)
mean_a = np.mean(tpr_a_all, axis=0)
ax.plot(grid, mean_b, '--', color=C_PRP, linewidth=1.0, alpha=0.7,
        label=f'Before ({MEAN_AUC_B:.3f})', zorder=4)
ax.plot(grid, mean_a, '-', color=C_PRP, linewidth=1.5,
        label=f'After ({MEAN_AUC_A:.3f})', zorder=5)
ax.fill(np.concatenate([grid, grid[::-1]]),
        np.concatenate([mean_a, mean_b[::-1]]), color=C_PRP, alpha=0.10,
        zorder=2)
ax.plot([0, 1], [0, 1], 'k:', linewidth=0.5, alpha=0.3)
ax.set_xlabel('FPR', fontsize=7)
ax.set_ylabel('TPR', fontsize=7)
ax.set_title(f'PRP-TCR CD69\n(per-TCR mean ΔAUROC {MEAN_DAUC:+.3f})',
             fontweight='bold', fontsize=7, color=C_PRP)
ax.text(0.97, 0.30, f'n={N_TCR} TCRs · per-TCR mean\n'
        f'(thin = each TCR; pooling N/A:\nper-patient score scales)',
        transform=ax.transAxes, fontsize=3.6, va='top', ha='right',
        color='#555', fontstyle='italic')
ax.legend(fontsize=5, loc='lower right')
ax.tick_params(labelsize=6)
save(fig, 'prp', 'roc')


# ═══════════════════════════════════════════
# Panel 3: TDR improvement (Panel-H/J twin-axis format)
# Σ-over-TCR discovered activators in each TCR's top-k
# ═══════════════════════════════════════════
print("\n=== PRP-TCR TDR improvement ===")
fig, ax = plt.subplots(figsize=(PW, PH))
ks = [1, 3, 5]
cum_raw, cum_cal = [], []
for k in ks:
    rr_k = sum(int(g.sort_values('raw_rank_in_tcr').head(k)['y_true'].sum())
               for _, g in s.groupby('tcr_id'))
    cc_k = sum(int(g.sort_values('cal_rank_in_tcr').head(k)['y_true'].sum())
               for _, g in s.groupby('tcr_id'))
    cum_raw.append(rr_k)
    cum_cal.append(cc_k)
denom = [N_TCR * k for k in ks]
tdr_raw = [cum_raw[i] / denom[i] for i in range(len(ks))]
tdr_cal = [cum_cal[i] / denom[i] for i in range(len(ks))]

ax.plot(ks, tdr_raw, '--', color=C_PRP, linewidth=1.0, alpha=0.5,
        label='Model only (TDR)')
ax.plot(ks, tdr_cal, '-', color=C_PRP, linewidth=1.5,
        label='CaliPPer (TDR)')
ax.fill_between(ks, tdr_raw, tdr_cal, color=C_PRP, alpha=0.08)
ax.set_xlabel('Top-k (within each TCR)', fontsize=7)
ax.set_ylabel('True Discovery Rate', fontsize=7, color=C_PRP)
ax.tick_params(axis='y', labelcolor=C_PRP, labelsize=6)
ax.tick_params(axis='x', labelsize=6)
ax.set_xticks(ks)
ax.set_ylim(-0.02, 1.05)

ax2 = ax.twinx()
ax2.plot(ks, cum_raw, '--', color='#555555', linewidth=1.0, alpha=0.5,
         label='Model only (count)')
ax2.plot(ks, cum_cal, '-', color='#555555', linewidth=1.5,
         label='CaliPPer (count)')
ax2.fill_between(ks, cum_raw, cum_cal, color='#555555', alpha=0.06)
ax2.set_ylabel(f'Activators discovered (Σ over {N_TCR} TCRs)',
               fontsize=7, color='#555555')
ax2.tick_params(axis='y', labelcolor='#555555', labelsize=6)
ax2.set_ylim(0, max(cum_cal) + 2)

deltas = [tdr_cal[i] - tdr_raw[i] for i in range(len(ks))]
bi = max(range(len(deltas)), key=lambda i: deltas[i])
if deltas[bi] > 0.01:
    ax.annotate(f'{cum_raw[bi]}/{denom[bi]}→{cum_cal[bi]}/{denom[bi]}'
                f'  (ΣΔTDR@{ks[bi]} {cum_cal[bi]-cum_raw[bi]:+d})',
                xy=(ks[bi], tdr_cal[bi]),
                xytext=(0.40, 0.85), textcoords='axes fraction',
                fontsize=5, color=C_PRP, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_PRP,
                                linewidth=0.6))
ax.set_title('PRP-TCR CD69', fontweight='bold', fontsize=7, color=C_PRP)
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, fontsize=3.5, loc='lower right',
          framealpha=0.9)
save(fig, 'prp', 'tdr')

print(f"\n{'=' * 60}")
print("PRP-TCR panels generated (per-TCR level; PRP-TCR colour "
      f"{C_PRP}).")
print(f"Output: {OUT_DIR}/fig6_prp_{{reranking,roc,tdr}}.{{png,pdf}}")

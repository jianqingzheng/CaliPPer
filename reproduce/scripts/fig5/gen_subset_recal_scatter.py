#!/usr/bin/env python3
"""Generate fig5_subset_recal_scatter: per-epitope recalibration before/after scatter.

Shows TCR unseen epitopes with AUROC (red) and AP (blue) before vs after recalibration.
Bubble size = sample count. Points above diagonal = improved.

Usage:
    cd <published_repo>/CaliPPer
    PYTHONPATH=Manuscript/designed_figures:Manuscript/designed_figures/panels:. \
        python Manuscript/designed_figures/panels/fig5/scripts/gen_subset_recal_scatter.py
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
try:
    from style_config import apply_publication_style, DPI
    apply_publication_style()
except Exception as _e_style:
    import sys as _s_style
    print(f"  ⚠ FALLBACK [gen_subset_recal_scatter]: style_config import/apply failed ({type(_e_style).__name__}: {_e_style}); using DPI=200 default", file=_s_style.stderr, flush=True)
    DPI = 200

from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
TCR_CT_SETS = ['unseen_fold34']  # unseen only for cleaner signal
CAL_SETS = ['v3_combined', 'v4_combined']
METRICS = ['aucroc', 'ap']

results = []

for model in TCR_MODELS:
    # Load all test sets
    all_data = {}
    for ts in CAL_SETS + TCR_CT_SETS:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                 f'{ts}_predictions_with_label.csv')
        dist_suffix = DIST_SUFFIX.get(DIST_TYPE, '_dist.npy')
        dist_path = os.path.join(RESULTS, 'fig2_cache', f'{model}_ct_{ts}{dist_suffix}')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path)
        d = np.load(dist_path)[:len(df)]
        y_col = 'binder' if 'binder' in df.columns else 'y_true'
        p_col = 'prediction' if 'prediction' in df.columns else 'y_prob'
        ep_col = 'peptide' if 'peptide' in df.columns else 'Epitope'
        all_data[ts] = {
            'y': df[y_col].values.astype(int),
            'p': df[p_col].values.astype(float),
            'd': d,
            'ep': df[ep_col].astype(str).values,
        }

    # Cal = v3+v4, test = unseen
    if not all(ts in all_data for ts in CAL_SETS + TCR_CT_SETS):
        continue

    cal_y = np.concatenate([all_data[ts]['y'] for ts in CAL_SETS])
    cal_p = np.concatenate([all_data[ts]['p'] for ts in CAL_SETS])
    cal_d = np.concatenate([all_data[ts]['d'] for ts in CAL_SETS])

    cal_data = {ts: (all_data[ts]['y'], all_data[ts]['p'], all_data[ts]['d']) for ts in CAL_SETS}
    ppv_p, npv_p, p_pos, p_neg, cal_prev = fit_recalibration(cal_data)

    for ts in TCR_CT_SETS:
        test = all_data[ts]
        cs = apply_recalibration(test['y'], test['p'], test['d'], ppv_p, npv_p, p_pos, p_neg, prev=cal_prev)

        # Per-epitope metrics
        for ep in np.unique(test['ep']):
            mask = test['ep'] == ep
            n = mask.sum()
            if n < 30:
                continue
            for metric in METRICS:
                before = safe_metric(metric, test['y'][mask], test['p'][mask])
                after = safe_metric(metric, test['y'][mask], cs[mask])
                if not np.isnan(before) and not np.isnan(after):
                    results.append(dict(model=model, epitope=ep, metric=metric,
                                        before=before, after=after, n=n, dist=test['d'][mask].mean()))

df = pd.DataFrame(results)
if len(df) == 0:
    print("No results generated")
    sys.exit(1)

# Plot
fig, ax = plt.subplots(figsize=(3.0, 3.0))
for metric, color, marker in [('aucroc', '#C44E52', 'o'), ('ap', '#4C72B0', 'o')]:
    sub = df[df['metric'] == metric]
    n_imp = (sub['after'] > sub['before']).sum()
    sizes = np.clip(sub['n'].values / 50, 3, 60)
    ax.scatter(sub['before'], sub['after'], c=color, s=sizes, alpha=0.4,
               edgecolor='white', linewidth=0.1, label=f'{metric.upper()} ({n_imp}/{len(sub)})')

lim = [0, 1]
ax.plot(lim, lim, 'k--', linewidth=0.5, alpha=0.3)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Metric before recalibration', fontsize=7)
ax.set_ylabel('Metric after recalibration', fontsize=7)
ax.set_title('Per-epitope recalibration\n(TCR unseen)', fontweight='bold', fontsize=8)
ax.legend(fontsize=5, loc='lower right', title='Metric (improved)', title_fontsize=4)
ax.tick_params(labelsize=6)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Size legend
from matplotlib.lines import Line2D
size_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(200/50),
           label='n=200'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(2000/50),
           label='n=2k'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(8000/50),
           label='n=8k'),
]
ax2 = ax.legend(handles=size_handles, fontsize=4, loc='upper left', title='Sample count',
                title_fontsize=4, framealpha=0.9)
ax.add_artist(ax2)

# Save
out_dir = os.path.join(FIG_DIR, 'fig5')
for subdir in ['blosum-sqrt', 'lev-logtransf']:
    d = os.path.join(out_dir, subdir)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, 'fig5_subset_recal_scatter.png'), dpi=DPI, bbox_inches='tight')
    fig.savefig(os.path.join(d, 'fig5_subset_recal_scatter.pdf'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved fig5_subset_recal_scatter ({len(df)} points)")

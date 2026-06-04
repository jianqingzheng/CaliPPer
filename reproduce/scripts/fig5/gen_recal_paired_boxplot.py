#!/usr/bin/env python3
"""Fig 5: Paired boxplots — before vs after recalibration by domain.

Two panel sets:
  1. Dataset-level: one point per (model × test_set/domain)
     - TCR: 5 models × 4 test sets = 20 points, grouped by seen/unseen
     - BCR: 5 models × 2 domains = 10 points (pooled AUROC per domain)
  2. Subset-level: one point per (model × epitope/variant)
     - TCR: per-epitope (≥30 samples)
     - BCR: per-variant LOO (≥30 samples)

Each group: paired boxes (before/after) with data points connected by thin lines.
Significance: Wilcoxon signed-rank test.

Data:
  TCR: v3+v4 cal, sigma_C distances, 4 test sets
  BCR: fold4cal pipeline, per-domain pooled or per-variant LOO
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
_FIG_DIR = os.path.join(FIG_DIR, 'fig5')
os.makedirs(_FIG_DIR, exist_ok=True)
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration
from style_config import apply_publication_style, MODEL_COLORS, DPI
from dist_config import DIST_TYPE, DIST_SUFFIX, DIST_SUBDIR, BCR_DIST_MODE, get_bcr_ct_distance

# Per-domain distance: BLOSUM for TCR (≤20 AA), Lev for BCR (≥100 AA)
TCR_DIST_SUFFIX = DIST_SUFFIX['blosum-sqrt']
BCR_DIST_MODE_ACTUAL = BCR_DIST_MODE['lev-log']

PANEL_DIR = os.path.join(_FIG_DIR, DIST_SUBDIR[DIST_TYPE])
os.makedirs(PANEL_DIR, exist_ok=True)

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_DIST_CACHE = os.path.join(RESULTS, 'fig2_cache')
BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
TCR_CAL_SETS = ['v3_combined', 'v4_combined']
TCR_TEST_SETS = ['seen_test', 'unseen_fold34', 'mcpas', 'iedb_sars']
TCR_SEEN = {'seen_test'}
MIN_SAMPLES = 30

MODEL_COLORS_ALL = {**MODEL_COLORS,
                    'xbcr': '#1f77b4', 'deepaai': '#ff7f0e',
                    'mambaaai': '#2ca02c', 'mint': '#d62728', 'rleaai': '#9467bd'}
MODEL_DISPLAY_ALL = {
    'nettcr': 'NetTCR', 'atm_tcr': 'ATM-TCR', 'blosum_rf': 'BLOSUM-RF',
    'ergo_ii': 'ERGO-II', 'tcrbert': 'TCR-BERT',
    'xbcr': 'XBCR-net', 'deepaai': 'DeepAAI', 'mambaaai': 'MambaAAI',
    'mint': 'MINT', 'rleaai': 'RLEAAI',
}


def save(fig, name):
    for ext, d in [('pdf', 300), ('png', 200)]:
        fig.savefig(os.path.join(PANEL_DIR, f'{name}.{ext}'), dpi=d, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


# ═══════════════════════════════════════════
# Compute TCR recalibration
# ═══════════════════════════════════════════
print("Computing TCR recalibration...")
records_dataset = []  # (model, group, metric, before, after, domain)
records_subset = []   # (model, group, metric, before, after, domain, subset_name, n)

for model in TCR_MODELS:
    ct = {}
    for ts in TCR_CAL_SETS + TCR_TEST_SETS:
        pp = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                          f'{ts}_predictions_with_label.csv')
        dp = os.path.join(TCR_DIST_CACHE, f'{model}_ct_{ts}{TCR_DIST_SUFFIX}')
        if not os.path.exists(pp) or not os.path.exists(dp):
            continue
        df = pd.read_csv(pp)
        d = np.load(dp)
        n = min(len(d), len(df))
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
        ep_col = 'peptide' if 'peptide' in df.columns else 'Epitope'
        ct[ts] = {'y': df[lc].values[:n].astype(int),
                  'p': df[pc].values[:n].astype(float),
                  'd': d[:n].astype(float),
                  'ep': df[ep_col].values[:n] if ep_col in df.columns else None}

    if not all(s in ct for s in TCR_CAL_SETS):
        continue

    cal_data = {s: (ct[s]['y'], ct[s]['p'], ct[s]['d']) for s in TCR_CAL_SETS}
    ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data)

    for ts in TCR_TEST_SETS:
        if ts not in ct:
            continue
        test = ct[ts]
        cal_s = apply_recalibration(test['y'], test['p'], test['d'],
                                    ppv_p, npv_p, pp, pn, prev=_cal_prev)
        group = 'TCR seen' if ts in TCR_SEEN else 'TCR unseen'

        # Dataset-level
        for metric in ['aucroc', 'ap']:
            before = safe_metric(metric, test['y'], test['p'])
            after = safe_metric(metric, test['y'], cal_s)
            records_dataset.append((model, group, metric, before, after, 'TCR'))

        # Per-epitope subset-level
        if test['ep'] is not None:
            for ep in np.unique(test['ep']):
                mask = test['ep'] == ep
                if mask.sum() < MIN_SAMPLES:
                    continue
                yi = test['y'][mask]
                if yi.sum() == 0 or yi.sum() == len(yi):
                    continue
                for metric in ['aucroc', 'ap']:
                    b = safe_metric(metric, yi, test['p'][mask])
                    a = safe_metric(metric, yi, cal_s[mask])
                    if not np.isnan(b) and not np.isnan(a):
                        records_subset.append((model, group, metric, b, a, 'TCR', ep, mask.sum()))

print(f"  TCR dataset: {len([r for r in records_dataset if r[5]=='TCR'])} records")
print(f"  TCR subset: {len([r for r in records_subset if r[5]=='TCR'])} records")


# ═══════════════════════════════════════════
# Compute BCR recalibration
# ═══════════════════════════════════════════
print("Computing BCR recalibration...")

for model in BCR_MODELS:
    model_dir = os.path.join(BCR_FOLD4CAL, model)
    if not os.path.isdir(model_dir):
        continue
    cal_path = os.path.join(model_dir, "cal_predictions.csv")

    if not os.path.exists(cal_path):

        print(f"  [BCR] {model}: cal_predictions.csv missing; skipping")

        continue

    cal = pd.read_csv(cal_path)
    if BCR_DIST_MODE_ACTUAL == 'npy_sidecar':
        cal['distance'] = get_bcr_ct_distance(cal, model_dir, 'cal_predictions')
    parts = [cal]
    for ts in ['A1-A11', 'unseen', 'flu']:
        fp = os.path.join(model_dir, f'{ts}_predictions.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if BCR_DIST_MODE_ACTUAL == 'npy_sidecar':
            df['distance'] = get_bcr_ct_distance(df, model_dir, ts)
        if 'data_source' not in df.columns:
            df['data_source'] = 'flu' if ts == 'flu' else 'sars'
        parts.append(df)
    pooled = pd.concat(parts, ignore_index=True)

    for domain in ['sars', 'flu']:
        domain_df = pooled[pooled['data_source'] == domain]
        group = f'BCR {domain.upper()}'

        # Per-variant LOO within domain (matching generate_fig5_new_panels.py)
        variants = domain_df.groupby('variant_seq').size()
        valid_v = variants[variants >= MIN_SAMPLES].index.tolist()
        all_before_y, all_before_p, all_after_p = [], [], []
        for held_v in valid_v:
            t_mask = domain_df['variant_seq'] == held_v
            c_mask = ~t_mask
            cs = domain_df[c_mask]
            ts_df = domain_df[t_mask]
            ty = ts_df['rbd'].values.astype(int)
            if ty.sum() == 0 or ty.sum() == len(ty):
                continue
            cy = cs['rbd'].values.astype(int)
            if cy.sum() < 3 or (len(cy) - cy.sum()) < 3:
                continue
            cal_d = {'cal': (cy, cs['pred_prob'].values.astype(float),
                             cs['distance'].values.astype(float))}
            ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_d)
            cal_s = apply_recalibration(
                ty, ts_df['pred_prob'].values.astype(float),
                ts_df['distance'].values.astype(float),
                ppv_p, npv_p, pp, pn)
            all_before_y.extend(ty)
            all_before_p.extend(ts_df['pred_prob'].values.astype(float))
            all_after_p.extend(cal_s)

            # Subset-level record per variant
            for metric in ['aucroc', 'ap']:
                b = safe_metric(metric, ty, ts_df['pred_prob'].values.astype(float))
                a = safe_metric(metric, ty, cal_s)
                if not np.isnan(b) and not np.isnan(a):
                    records_subset.append((model, group, metric, b, a, 'BCR', held_v[:20], len(ty)))

        # Dataset-level from pooled LOO predictions
        if len(all_before_y) > 30:
            all_before_y = np.array(all_before_y)
            all_before_p = np.array(all_before_p)
            all_after_p = np.array(all_after_p)
            for metric in ['aucroc', 'ap']:
                before = safe_metric(metric, all_before_y, all_before_p)
                after = safe_metric(metric, all_before_y, all_after_p)
                if not np.isnan(before) and not np.isnan(after):
                    records_dataset.append((model, group, metric, before, after, 'BCR'))

print(f"  BCR dataset: {len([r for r in records_dataset if r[5]=='BCR'])} records")
print(f"  BCR subset: {len([r for r in records_subset if r[5]=='BCR'])} records")

df_ds = pd.DataFrame(records_dataset, columns=['model', 'group', 'metric', 'before', 'after', 'domain'])
df_ds['delta'] = df_ds['after'] - df_ds['before']

df_sub = pd.DataFrame(records_subset, columns=['model', 'group', 'metric', 'before', 'after', 'domain', 'subset', 'n'])
df_sub['delta'] = df_sub['after'] - df_sub['before']


# ═══════════════════════════════════════════
# Plotting function
# ═══════════════════════════════════════════
GROUPS = ['TCR seen', 'TCR unseen', 'BCR SARS', 'BCR FLU']
GROUP_COLORS = {
    'TCR seen': '#4DBEEE',
    'TCR unseen': '#0072BD',
    'BCR SARS': '#EDB120',
    'BCR FLU': '#D95319',
}


def plot_paired_boxplot(data, metric, level_label, filename):
    metric_label = 'AUROC' if metric == 'aucroc' else 'AP'
    sub = data[data['metric'] == metric].copy()

    # One narrow subplot per group
    active_groups = [g for g in GROUPS if len(sub[sub['group'] == g]) >= 2]
    n_groups = len(active_groups)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(1.2 * n_groups + 0.4, 3.8),
                              sharey=True, gridspec_kw={'wspace': 0.05})
    if n_groups == 1:
        axes = [axes]

    for gi, (ax, group) in enumerate(zip(axes, active_groups)):
        gsub = sub[sub['group'] == group].reset_index(drop=True)
        color = GROUP_COLORS[group]
        x_b, x_a = 0.35, 0.65

        # Connected lines — colored by direction
        for i in range(len(gsub)):
            delta_i = gsub.loc[i, 'after'] - gsub.loc[i, 'before']
            lc = '#66aa66' if delta_i > 0 else '#cc6666'
            ax.plot([x_b, x_a], [gsub.loc[i, 'before'], gsub.loc[i, 'after']],
                    color=lc, linewidth=0.8, alpha=0.55, zorder=2)

        # Data points colored by model
        rng = np.random.RandomState(gi * 7 + 3)
        jit_b = rng.uniform(-0.04, 0.04, len(gsub))
        jit_a = rng.uniform(-0.04, 0.04, len(gsub))
        for i in range(len(gsub)):
            mc = MODEL_COLORS_ALL.get(gsub.loc[i, 'model'], '#888')
            ax.scatter(x_b + jit_b[i], gsub.loc[i, 'before'], color=mc,
                       s=16, alpha=0.75, zorder=4, edgecolors='white', linewidths=0.3)
            ax.scatter(x_a + jit_a[i], gsub.loc[i, 'after'], color=mc,
                       s=16, alpha=0.75, zorder=4, edgecolors='white', linewidths=0.3)

        # Box plots — show mean line instead of median, lowlighted
        for pos, vals, alpha in [(x_b, gsub['before'].values, 0.12),
                                  (x_a, gsub['after'].values, 0.30)]:
            bp = ax.boxplot([vals], positions=[pos], widths=0.18,
                       patch_artist=True, showfliers=False, zorder=1,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor=color, alpha=alpha, edgecolor=color, linewidth=0.6),
                       medianprops=dict(linewidth=0),  # hide median
                       meanprops=dict(color='#999999', linewidth=1.0, linestyle='-'),
                       whiskerprops=dict(color=color, linewidth=0.6),
                       capprops=dict(color=color, linewidth=0.6))

        # Significance (Wilcoxon)
        try:
            deltas = gsub['after'].values - gsub['before'].values
            if np.all(deltas == 0):
                raise ValueError
            _, pval = wilcoxon(deltas, alternative='greater')
            sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))

            y_top = max(gsub['before'].max(), gsub['after'].max())
            y_bot = min(gsub['before'].min(), gsub['after'].min())
            y_range = y_top - y_bot

            mean_d = gsub['delta'].mean()

            # Place annotation above for TCR, below for BCR (avoid collision)
            # Two lines: Δ value farther from data, significance closer to bracket
            if 'BCR' in group:
                y_ann = y_bot - y_range * 0.12
                ax.plot([x_b, x_b, x_a, x_a],
                        [y_ann + y_range * 0.015, y_ann, y_ann, y_ann + y_range * 0.015],
                        color='black', linewidth=0.7)
                # sig closer to bracket
                ax.text(0.5, y_ann - y_range * 0.03, sig,
                        ha='center', va='top', fontsize=7.5, fontweight='bold')
                # Δ farther from bracket — more gap to avoid collision with sig
                ax.text(0.5, y_ann - y_range * 0.20,
                        f'Δ={mean_d:+.3f}',
                        ha='center', va='top', fontsize=6.5, color='#444444')
            else:
                y_ann = y_top + y_range * 0.05
                ax.plot([x_b, x_b, x_a, x_a],
                        [y_ann - y_range * 0.015, y_ann, y_ann, y_ann - y_range * 0.015],
                        color='black', linewidth=0.7)
                # sig closer to bracket
                ax.text(0.5, y_ann + y_range * 0.02, sig,
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')
                # Δ farther from bracket
                ax.text(0.5, y_ann + y_range * 0.11,
                        f'Δ={mean_d:+.3f}',
                        ha='center', va='bottom', fontsize=6.5, color='#444444')
        except (ValueError, Exception) as _e_ann:
            import sys as _s_ann
            print(f"  ⚠ FALLBACK [recal-boxplot annotation]: bracket/Δ-label rendering failed ({type(_e_ann).__name__}: {_e_ann}); skipping annotation", file=_s_ann.stderr, flush=True)

        # Formatting
        ax.set_xticks([x_b, x_a])
        ax.set_xticklabels(['Before', 'After'], fontsize=6.5, rotation=35, ha='right')
        ax.set_title(group, fontsize=8, fontweight='bold', pad=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if gi > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)
        else:
            ax.set_ylabel(metric_label, fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlim(0.15, 0.85)

    fig.suptitle(f'{level_label} {metric_label}', fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, filename)


# ═══════════════════════════════════════════
# Generate panels
# ═══════════════════════════════════════════
print("\nGenerating dataset-level panels...")
for metric in ['aucroc', 'ap']:
    plot_paired_boxplot(df_ds, metric, 'Dataset-level',
                        f'fig5_recal_paired_boxplot_dataset_{metric}')

print("\nGenerating subset-level panels...")
for metric in ['aucroc', 'ap']:
    plot_paired_boxplot(df_sub, metric, 'Subset-level',
                        f'fig5_recal_paired_boxplot_subset_{metric}')


# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print("\n=== Dataset-level summary ===")
for metric in ['aucroc', 'ap']:
    ml = 'AUROC' if metric == 'aucroc' else 'AP'
    for group in GROUPS:
        g = df_ds[(df_ds['metric'] == metric) & (df_ds['group'] == group)]
        if len(g) < 2:
            continue
        try:
            _, p = wilcoxon(g['delta'].values, alternative='greater')
        except Exception as _e_wx:
            import sys as _s_wx
            print(f"  ⚠ FALLBACK [recal-boxplot wilcoxon]: metric={metric} group={group} wilcoxon failed ({type(_e_wx).__name__}: {_e_wx}); using p=1.0", file=_s_wx.stderr, flush=True)
            p = 1.0
        n_pos = (g['delta'] > 0).sum()
        print(f"  {ml} {group:12s}: Δ={g['delta'].mean():+.3f} p={p:.1e} "
              f"({n_pos}/{len(g)} improved) n={len(g)}")

print("\n=== Subset-level summary ===")
for metric in ['aucroc', 'ap']:
    ml = 'AUROC' if metric == 'aucroc' else 'AP'
    for group in GROUPS:
        g = df_sub[(df_sub['metric'] == metric) & (df_sub['group'] == group)]
        if len(g) < 2:
            continue
        try:
            _, p = wilcoxon(g['delta'].values, alternative='greater')
        except Exception as _e_wx:
            import sys as _s_wx
            print(f"  ⚠ FALLBACK [recal-boxplot wilcoxon]: metric={metric} group={group} wilcoxon failed ({type(_e_wx).__name__}: {_e_wx}); using p=1.0", file=_s_wx.stderr, flush=True)
            p = 1.0
        n_pos = (g['delta'] > 0).sum()
        print(f"  {ml} {group:12s}: Δ={g['delta'].mean():+.3f} p={p:.1e} "
              f"({n_pos}/{len(g)} improved) n={len(g)}")

print("\nDone.")

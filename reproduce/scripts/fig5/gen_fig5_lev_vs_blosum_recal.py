#!/usr/bin/env python3
"""Fig 5 panel O: Lev vs BLOSUM recalibration ΔAUROC comparison.

Paired boxplots with thin lines connecting the same (model, test_set) pair.
Two groups: TCR (left) and BCR (right).
TCR: fixed-cal (v3+v4), 5 models × 4 test sets = 20 pairs.
BCR: per-variant LOO, 5 models × pooled SARS+FLU = 5 pairs.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
_FIG_DIR = os.path.join(FIG_DIR, 'fig5')
os.makedirs(_FIG_DIR, exist_ok=True)
from style_config import apply_publication_style
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
CACHE = os.path.join(RESULTS, 'fig2_cache')
FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
TCR_CAL = ['v3_combined', 'v4_combined']
TCR_TEST = ['seen_test', 'unseen_fold34', 'mcpas', 'iedb_sars']
MIN_SAMPLES = 30

LEV_COLOR = '#1f77b4'
BLOSUM_COLOR = '#ff7f0e'


def save(fig, name):
    for subdir in ['blosum-sqrt', 'lev-logtransf']:
        out_dir = os.path.join(_FIG_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name + '.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


# ═══════════════════════════════════════════
# TCR: fixed-cal, per (model, test_set) ΔAUROC
# ═══════════════════════════════════════════
print("=== TCR: Lev vs BLOSUM recalibration ===")

tcr_lev, tcr_blo = [], []  # paired lists

for model in TCR_MODELS:
    for dist_label, suffix in [('lev', '_dist.npy'), ('blosum', '_blosumsqrt_dist.npy')]:
        ct = {}
        for ts in TCR_CAL + TCR_TEST:
            pred = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                f'{ts}_predictions_with_label.csv')
            dist = os.path.join(CACHE, f'{model}_ct_{ts}{suffix}')
            if not os.path.exists(pred) or not os.path.exists(dist):
                continue
            df = pd.read_csv(pred)
            d = np.load(dist)
            n = min(len(d), len(df))
            lc = 'binder' if 'binder' in df.columns else 'y_true'
            pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
            ct[ts] = (df[lc].values[:n].astype(int), df[pc].values[:n].astype(float), d[:n])

        if not all(s in ct for s in TCR_CAL):
            continue
        cal_data = {s: ct[s] for s in TCR_CAL}
        ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data)

        for ts in TCR_TEST:
            if ts not in ct:
                continue
            y, p, d = ct[ts]
            cal_s = apply_recalibration(y, p, d, ppv_p, npv_p, pp, pn, prev=_cal_prev)
            delta = safe_metric('aucroc', y, cal_s) - safe_metric('aucroc', y, p)
            if dist_label == 'lev':
                tcr_lev.append(delta)
            else:
                tcr_blo.append(delta)

print(f"  TCR CT pairs: {len(tcr_lev)} Lev, {len(tcr_blo)} BLOSUM")

# TCR CV halfsplit (Lev and BLOSUM)
print("  TCR CV halfsplit...")
for model in TCR_MODELS:
    for fold in range(5):
        pred = os.path.join(RESULTS, model, 'cv_logdist',
                            f'fold{fold}', 'test_predictions_with_label.csv')
        if not os.path.exists(pred):
            continue
        df = pd.read_csv(pred)
        n_pred = len(df)
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'

        lev_deltas = {}
        for dist_label, suffix in [('lev', '_dist.npy'), ('blosum', '_blosumsqrt_dist.npy')]:
            dist_path = os.path.join(CACHE, f'{model}_cv_fold{fold}{suffix}')
            if not os.path.exists(dist_path):
                continue
            d = np.load(dist_path)
            n = min(len(d), n_pred)
            y = df[lc].values[:n].astype(int)
            p = df[pc].values[:n].astype(float)
            d = d[:n]
            si = np.argsort(d)
            cal_idx = si[0::2]; test_idx = si[1::2]
            if len(cal_idx) < 50:
                continue
            cal_data_cv = {'cal': (y[cal_idx], p[cal_idx], d[cal_idx])}
            ppv_p, npv_p, pp, pn, _cp = fit_recalibration(cal_data_cv)
            cs = apply_recalibration(y[test_idx], p[test_idx], d[test_idx],
                                      ppv_p, npv_p, pp, pn, prev=_cp)
            delta = safe_metric('aucroc', y[test_idx], cs) - safe_metric('aucroc', y[test_idx], p[test_idx])
            lev_deltas[dist_label] = delta

        if 'lev' in lev_deltas:
            tcr_lev.append(lev_deltas['lev'])
        if 'blosum' in lev_deltas:
            tcr_blo.append(lev_deltas['blosum'])

print(f"  TCR total pairs (CT+CV): {len(tcr_lev)} Lev, {len(tcr_blo)} BLOSUM")
print(f"  Lev mean: {np.mean(tcr_lev):+.4f}, BLOSUM mean: {np.mean(tcr_blo):+.4f}")

# ═══════════════════════════════════════════
# BCR: per-variant LOO, per model pooled ΔAUROC
# ═══════════════════════════════════════════
print("\n=== BCR: Lev vs BLOSUM recalibration ===")

bcr_lev, bcr_blo = [], []

for model in BCR_MODELS:
    model_dir = os.path.join(FOLD4CAL, model)
    cal_path = os.path.join(model_dir, 'cal_predictions.csv')
    if not os.path.exists(cal_path):
        continue
    cal_df = pd.read_csv(cal_path)
    if 'data_source' not in cal_df.columns:
        cal_df['data_source'] = 'sars'
    cal_df['source'] = 'fold4_test'

    for dist_label in ['lev', 'blosum']:
        parts = [cal_df.copy()]

        # Set distance column based on distance type
        if dist_label == 'blosum':
            blo_cal = os.path.join(model_dir, 'cal_predictions_blosumsqrt_dist.npy')
            if not os.path.exists(blo_cal):
                continue
            parts[0]['dist_use'] = np.load(blo_cal)[:len(cal_df)]
        else:
            if 'distance' not in cal_df.columns:
                continue
            parts[0]['dist_use'] = cal_df['distance'].values

        for ts in ['A1-A11', 'unseen', 'flu']:
            fp = os.path.join(model_dir, f'{ts}_predictions.csv')
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            df['source'] = ts
            if 'data_source' not in df.columns:
                df['data_source'] = 'flu' if ts == 'flu' else 'sars'
            if dist_label == 'blosum':
                blo_test = os.path.join(model_dir, f'{ts}_blosumsqrt_dist.npy')
                if not os.path.exists(blo_test):
                    continue
                df['dist_use'] = np.load(blo_test)[:len(df)]
            else:
                if 'distance' not in df.columns:
                    continue
                df['dist_use'] = df['distance'].values
            parts.append(df)

        pooled = pd.concat(parts, ignore_index=True)
        if 'dist_use' not in pooled.columns:
            continue

        for domain in ['sars', 'flu']:
            dom = pooled[pooled['data_source'] == domain]
            variants = dom.groupby('variant_seq').size()
            valid_v = variants[variants >= MIN_SAMPLES].index.tolist()
            all_y, all_raw, all_cal_s = [], [], []
            for held_v in valid_v:
                test_mask = dom['variant_seq'] == held_v
                cal_mask = ~test_mask
                cal_sub = dom[cal_mask]
                test_sub = dom[test_mask]
                if len(test_sub) < 10:
                    continue
                ty = test_sub['rbd'].values.astype(int)
                if ty.sum() == 0 or ty.sum() == len(ty):
                    continue
                cy = cal_sub['rbd'].values.astype(int)
                if cy.sum() < 3 or (len(cy) - cy.sum()) < 3:
                    continue
                cal_data_v = {'cal': (cy, cal_sub['pred_prob'].values.astype(float),
                                      cal_sub['dist_use'].values.astype(float))}
                tp = test_sub['pred_prob'].values.astype(float)
                td = test_sub['dist_use'].values.astype(float)
                ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data_v)
                cs = apply_recalibration(ty, tp, td, ppv_p, npv_p, pp, pn)
                all_y.extend(ty); all_raw.extend(tp); all_cal_s.extend(cs)

            if all_y:
                ay, ar, ac = np.array(all_y), np.array(all_raw), np.array(all_cal_s)
                delta = safe_metric('aucroc', ay, ac) - safe_metric('aucroc', ay, ar)
                if dist_label == 'lev':
                    bcr_lev.append(delta)
                else:
                    bcr_blo.append(delta)

print(f"  BCR CT pairs: {len(bcr_lev)} Lev, {len(bcr_blo)} BLOSUM")

print(f"  BCR CT pairs (per-domain): {len(bcr_lev)} Lev, {len(bcr_blo)} BLOSUM")
print(f"  Lev mean: {np.mean(bcr_lev):+.4f}, BLOSUM mean: {np.mean(bcr_blo):+.4f}")

# ═══════════════════════════════════════════
# Plot: two subplots — TCR (CT+CV) left, BCR (CT only) right
# ═══════════════════════════════════════════
fig, (ax_tcr, ax_bcr) = plt.subplots(1, 2, figsize=(4.5, 3.0),
                                      gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.35})

pos = [0, 1]  # Lev=0, BLOSUM=1

def plot_paired_box(ax, lev_data, blo_data, title, seed=42):
    """Plot paired boxplots with connecting lines on given axis."""
    bp = ax.boxplot([lev_data, blo_data], positions=pos, widths=0.6,
                     patch_artist=True, showfliers=False, showmeans=True,
                     meanline=True, meanprops=dict(color='black', linewidth=1.5),
                     medianprops=dict(linewidth=0))
    bp['boxes'][0].set_facecolor(LEV_COLOR); bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor(BLOSUM_COLOR); bp['boxes'][1].set_alpha(0.4)

    # Paired lines
    for lv, bl in zip(lev_data, blo_data):
        ax.plot(pos, [lv, bl], '-', color='gray', alpha=0.25, linewidth=0.5)

    # Dots
    rng = np.random.default_rng(seed)
    j1 = rng.uniform(-0.1, 0.1, len(lev_data))
    j2 = rng.uniform(-0.1, 0.1, len(blo_data))
    ax.scatter(np.full(len(lev_data), pos[0]) + j1, lev_data,
               c=LEV_COLOR, s=15, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)
    ax.scatter(np.full(len(blo_data), pos[1]) + j2, blo_data,
               c=BLOSUM_COLOR, s=15, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

    # Mean annotations (defensive: if either array is empty, skip the
    # annotation to avoid ValueError: max() arg is an empty sequence —
    # happens when BCR cal_predictions.csv is absent from the deposit).
    if len(lev_data) > 0:
        ax.text(pos[0], max(lev_data) + 0.015, f'{np.mean(lev_data):+.3f}', ha='center',
                fontsize=6, fontweight='bold', color=LEV_COLOR)
    if len(blo_data) > 0:
        ax.text(pos[1], max(blo_data) + 0.015, f'{np.mean(blo_data):+.3f}', ha='center',
                fontsize=6, fontweight='bold', color=BLOSUM_COLOR)

    ax.set_xticks(pos)
    ax.set_xticklabels(['Lev', 'BLOSUM'], fontsize=7)
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# TCR: CT + CV (45 pairs)
plot_paired_box(ax_tcr, tcr_lev, tcr_blo, f'TCR (CT+CV, n={len(tcr_lev)})', seed=42)
ax_tcr.set_ylabel('ΔAUROC', fontsize=9)

# BCR: CT only per-domain (10 pairs)
plot_paired_box(ax_bcr, bcr_lev, bcr_blo, f'BCR (CT, n={len(bcr_lev)})', seed=44)

# Shared legend
ax_tcr.legend([Patch(facecolor=LEV_COLOR, alpha=0.4),
               Patch(facecolor=BLOSUM_COLOR, alpha=0.4)],
              ['Levenshtein', 'BLOSUM-SW'], fontsize=6, loc='upper right', framealpha=0.8)

save(fig, 'fig5_lev_vs_blosum_recal')
print("\nDone.")

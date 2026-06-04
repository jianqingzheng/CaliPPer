#!/usr/bin/env python3
"""Generate additional fig5 panels for reorganized layout.

New panels:
  - fig5_perbin_delta_bcr: BCR far-samples-gain-most (Lev distances)
  - fig5_recal_paired_boxplot_dataset_ap: dataset-level AP boxplot
  - fig5_auroc_vs_ppvnpv_mae_tcr: ΔAUROC vs PPV/NPV MAE scatter (TCR)
  - fig5_auroc_vs_ppvnpv_mae_bcr: ΔAUROC vs PPV/NPV MAE scatter (BCR)
  - fig5_placeholder_concept: placeholder for concept diagrams

Uses BLOSUM for TCR, Lev for BCR (per-domain distance rule).
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
_FIG_DIR = os.path.join(FIG_DIR, 'fig5')
os.makedirs(_FIG_DIR, exist_ok=True)
from style_config import apply_publication_style, DPI
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')
FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
TCR_DISPLAY = {'nettcr': 'NetTCR', 'atm_tcr': 'ATM-TCR', 'blosum_rf': 'BLOSUM-RF',
               'ergo_ii': 'ERGO-II', 'tcrbert': 'TCR-BERT'}
BCR_DISPLAY = {'xbcr': 'XBCR-net', 'deepaai': 'DeepAAI', 'mambaaai': 'MambaAAI',
               'mint': 'MINT', 'rleaai': 'RLEAAI'}
MODEL_COLORS = {'nettcr': '#1f77b4', 'atm_tcr': '#ff7f0e', 'blosum_rf': '#2ca02c',
                'ergo_ii': '#d62728', 'tcrbert': '#9467bd'}
BCR_MODEL_COLORS = {'xbcr': '#1f77b4', 'deepaai': '#ff7f0e', 'mambaaai': '#2ca02c',
                     'mint': '#d62728', 'rleaai': '#9467bd'}

TCR_CAL = ['v3_combined', 'v4_combined']
TCR_TEST = ['seen_test', 'unseen_fold34', 'mcpas', 'iedb_sars']
N_BINS = 8
MIN_SAMPLES = 30


def save(fig, name):
    for subdir in ['blosum-sqrt', 'lev-logtransf']:
        out_dir = os.path.join(_FIG_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name + '.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


# ═══════════════════════════════════════════
# Load TCR data (BLOSUM distances)
# ═══════════════════════════════════════════
print("Loading TCR data (BLOSUM)...")
tcr_data = {}  # model -> {ts -> (y, p, d)}
for model in TCR_MODELS:
    tcr_data[model] = {}
    for ts in TCR_CAL + TCR_TEST:
        pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                  f'{ts}_predictions_with_label.csv')
        dist_path = os.path.join(TCR_CACHE, f'{model}_ct_{ts}_blosumsqrt_dist.npy')
        if not os.path.exists(pred_path) or not os.path.exists(dist_path):
            continue
        df = pd.read_csv(pred_path)
        d = np.load(dist_path)
        n = min(len(d), len(df))
        lc = 'binder' if 'binder' in df.columns else 'y_true'
        pc = 'prediction' if 'prediction' in df.columns else 'y_prob'
        tcr_data[model][ts] = (df[lc].values[:n].astype(int),
                                df[pc].values[:n].astype(float), d[:n])

# ═══════════════════════════════════════════
# Load BCR data (Lev distances from CSV column)
# ═══════════════════════════════════════════
print("Loading BCR data (Lev)...")
bcr_data = {}  # model -> {'cal': ..., 'A1-A11': ..., etc}
for model in BCR_MODELS:
    model_dir = os.path.join(FOLD4CAL, model)
    cal_path = os.path.join(model_dir, 'cal_predictions.csv')
    if not os.path.exists(cal_path):
        continue
    cal = pd.read_csv(cal_path)
    if 'distance' not in cal.columns:
        continue
    bcr_data[model] = {'cal': (cal['rbd'].values.astype(int),
                                cal['pred_prob'].values.astype(float),
                                cal['distance'].values.astype(float))}
    for ts in ['A1-A11', 'unseen', 'flu']:
        fp = os.path.join(model_dir, f'{ts}_predictions.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if 'distance' not in df.columns:
            continue
        bcr_data[model][ts] = (df['rbd'].values.astype(int),
                                df['pred_prob'].values.astype(float),
                                df['distance'].values.astype(float))


# ═══════════════════════════════════════════
# Panel: Placeholder concept (a/b)
# ═══════════════════════════════════════════
print("\nGenerating placeholder concept panels...")
for label in ['a', 'b']:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    ax.text(0.5, 0.5, f'Conceptual diagram\n(to be designed)', ha='center', va='center',
            fontsize=10, color='#aaaaaa', style='italic', transform=ax.transAxes)
    ax.set_facecolor('#fafafa')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#dddddd')
    save(fig, f'fig5_placeholder_{label}')


# ═══════════════════════════════════════════
# Panel M: BCR far-samples-gain-most (perbin delta)
# ═══════════════════════════════════════════
print("\nGenerating BCR perbin delta (far samples gain most)...")

bcr_perbin = []
for model in BCR_MODELS:
    if model not in bcr_data or 'cal' not in bcr_data[model]:
        continue
    # Per-variant LOO within domain
    model_dir = os.path.join(FOLD4CAL, model)
    cal_df = pd.read_csv(os.path.join(model_dir, 'cal_predictions.csv'))
    if 'data_source' not in cal_df.columns:
        cal_df['data_source'] = 'sars'
    cal_df['source'] = 'fold4_test'
    parts = [cal_df]
    for ts in ['A1-A11', 'unseen', 'flu']:
        fp = os.path.join(model_dir, f'{ts}_predictions.csv')
        if not os.path.exists(fp):
            import sys as _s_addpanel
            print(f"  ⚠ MISSING [fig5-additional]: ts={ts} not found at {fp}; skipping", file=_s_addpanel.stderr, flush=True)
            continue
        df = pd.read_csv(fp)
        df['source'] = ts
        if 'data_source' not in df.columns:
            df['data_source'] = 'flu' if ts == 'flu' else 'sars'
        parts.append(df)
    pooled = pd.concat(parts, ignore_index=True)
    if 'distance' not in pooled.columns:
        continue

    # Collect all recalibrated predictions
    all_y, all_p, all_cal, all_d = [], [], [], []
    for domain in ['sars', 'flu']:
        dom = pooled[pooled['data_source'] == domain]
        variants = dom.groupby('variant_seq').size()
        valid_v = variants[variants >= MIN_SAMPLES].index.tolist()
        for held_v in valid_v:
            test_mask = dom['variant_seq'] == held_v
            cal_mask = ~test_mask
            cal_sub = dom[cal_mask]
            test_sub = dom[test_mask]
            if len(test_sub) < 10: continue
            ty = test_sub['rbd'].values.astype(int)
            if ty.sum() == 0 or ty.sum() == len(ty): continue
            cy = cal_sub['rbd'].values.astype(int)
            if cy.sum() < 3 or (len(cy) - cy.sum()) < 3: continue
            cal_data_v = {'cal': (cy, cal_sub['pred_prob'].values.astype(float),
                                  cal_sub['distance'].values.astype(float))}
            tp = test_sub['pred_prob'].values.astype(float)
            td = test_sub['distance'].values.astype(float)
            ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data_v)
            cs = apply_recalibration(ty, tp, td, ppv_p, npv_p, pp, pn)
            all_y.extend(ty); all_p.extend(tp); all_cal.extend(cs); all_d.extend(td)

    if not all_y:
        continue
    all_y, all_p, all_cal, all_d = np.array(all_y), np.array(all_p), np.array(all_cal), np.array(all_d)

    # Per-bin delta
    si = np.argsort(all_d)
    bs = len(si) // N_BINS
    for bi in range(N_BINS):
        s = bi * bs
        e = len(si) if bi == N_BINS - 1 else (bi + 1) * bs
        idx = si[s:e]
        yi, pi, ci = all_y[idx], all_p[idx], all_cal[idx]
        if yi.sum() == 0 or yi.sum() == len(yi): continue
        ab = safe_metric('aucroc', yi, pi)
        aa = safe_metric('aucroc', yi, ci)
        bcr_perbin.append((model, bi, all_d[idx].mean(), aa - ab))

if bcr_perbin:
    df_bin = pd.DataFrame(bcr_perbin, columns=['model', 'bin', 'd_center', 'delta'])
    bin_avg = df_bin.groupby('bin').agg(d=('d_center', 'mean'), delta=('delta', 'mean')).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    colors_grad = plt.cm.Greens(np.linspace(0.3, 0.9, N_BINS))
    ax.bar(range(N_BINS), bin_avg['delta'], color=colors_grad, edgecolor='white', linewidth=0.5)

    # Trend line
    r, p = pearsonr(bin_avg.index, bin_avg['delta'])
    z = np.polyfit(bin_avg.index, bin_avg['delta'], 1)
    ax.plot(range(N_BINS), np.polyval(z, range(N_BINS)), '--', color='gray', linewidth=1)
    ax.text(0.05, 0.95, f'r = {r:.3f} (p = {p:.3f})', transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel('ΔAUROC', fontsize=9)
    ax.set_xlabel('Distance bin (near → far)', fontsize=8)
    ax.set_title('Far samples gain most (BCR)', fontweight='bold', fontsize=9)
    ax.set_xticks([0, N_BINS-1]); ax.set_xticklabels(['(near)', '(far)'])
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'fig5_perbin_delta_bcr')


# ═══════════════════════════════════════════
# Panels O/P: ΔAUROC vs PPV/NPV prediction MAE
# ═══════════════════════════════════════════
print("\nGenerating ΔAUROC vs PPV/NPV MAE scatter...")

def compute_ppvnpv_mae(y, p, d, ppv_params, npv_params, p_pos, p_neg, n_bins=8):
    """Compute MAE of PPV and NPV predictions vs actual per-bin values."""
    from calipper.core import _logit, _sig, adaptive_n_bins

    si = np.argsort(d)
    bs = max(len(si) // n_bins, 1)

    ppv_mae_list, npv_mae_list = [], []
    threshold = max(2*y.mean() - 1, min(2*y.mean(), 0.5))

    for i in range(n_bins):
        s = i * bs
        e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]
        yi, pi = y[idx], p[idx]
        di_mean = d[idx].mean()
        mp_mean = pi.mean()

        pred_pos = pi >= threshold
        if pred_pos.sum() > 0 and yi[pred_pos].size > 0:
            actual_ppv = yi[pred_pos].mean()
            # Predicted PPV from curve
            if ppv_params is not None:
                a, bx, c, beta = ppv_params
                pred_ppv = np.clip(a * np.exp(-bx * di_mean) + c + beta * mp_mean, 0.01, 0.99)
                ppv_mae_list.append(abs(pred_ppv - actual_ppv))

        pred_neg = ~pred_pos
        if pred_neg.sum() > 0 and yi[pred_neg].size > 0:
            actual_npv = 1.0 - yi[pred_neg].mean()
            if npv_params is not None:
                a, bx, c, beta = npv_params
                pred_npv = np.clip(a * np.exp(-bx * di_mean) + c + beta * mp_mean, 0.01, 0.99)
                npv_mae_list.append(abs(pred_npv - actual_npv))

    ppv_mae = np.mean(ppv_mae_list) if ppv_mae_list else np.nan
    npv_mae = np.mean(npv_mae_list) if npv_mae_list else np.nan
    return ppv_mae, npv_mae


# TCR
print("  TCR...")
tcr_scatter_data = []  # (model, ts, delta_auroc, ppv_mae, npv_mae)
for model in TCR_MODELS:
    if not all(s in tcr_data[model] for s in TCR_CAL):
        continue
    cal_data = {s: tcr_data[model][s] for s in TCR_CAL}
    ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data)

    for ts in TCR_TEST:
        if ts not in tcr_data[model]: continue
        y, p, d = tcr_data[model][ts]
        cal_s = apply_recalibration(y, p, d, ppv_p, npv_p, pp, pn, prev=_cal_prev)
        delta = safe_metric('aucroc', y, cal_s) - safe_metric('aucroc', y, p)
        ppv_mae, npv_mae = compute_ppvnpv_mae(y, p, d, ppv_p, npv_p, pp, pn)
        tcr_scatter_data.append((model, ts, delta, ppv_mae, npv_mae))

# BCR (per-variant LOO pooled)
print("  BCR...")
bcr_scatter_data = []
for model in BCR_MODELS:
    if model not in bcr_data or 'cal' not in bcr_data[model]:
        continue
    # Use fold4 cal for fit, apply to each test set
    cal_data = {'cal': bcr_data[model]['cal']}
    ppv_p, npv_p, pp, pn, _cal_prev = fit_recalibration(cal_data)

    for ts in ['A1-A11', 'unseen', 'flu']:
        if ts not in bcr_data[model]: continue
        y, p, d = bcr_data[model][ts]
        cal_s = apply_recalibration(y, p, d, ppv_p, npv_p, pp, pn)
        delta = safe_metric('aucroc', y, cal_s) - safe_metric('aucroc', y, p)
        ppv_mae, npv_mae = compute_ppvnpv_mae(y, p, d, ppv_p, npv_p, pp, pn)
        bcr_scatter_data.append((model, ts, delta, ppv_mae, npv_mae))

# Plot TCR
if tcr_scatter_data:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    for model, ts, delta, ppv_mae, npv_mae in tcr_scatter_data:
        c = MODEL_COLORS[model]
        if not np.isnan(ppv_mae):
            ax.scatter(ppv_mae, delta, c=c, marker='o', s=30, alpha=0.7,
                       edgecolors='white', linewidth=0.3)
        if not np.isnan(npv_mae):
            ax.scatter(npv_mae, delta, c=c, marker='s', s=30, alpha=0.7,
                       edgecolors='white', linewidth=0.3)

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='PPV MAE'),
               Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=6, label='NPV MAE')]
    for m in TCR_MODELS:
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=MODEL_COLORS[m],
                               markersize=5, label=TCR_DISPLAY[m]))
    ax.legend(handles=handles, fontsize=5, loc='lower right', framealpha=0.8, ncol=2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('PPV/NPV prediction MAE', fontsize=8)
    ax.set_ylabel('ΔAUROC', fontsize=8)
    ax.set_title('Recalibration gain vs\nPPV/NPV accuracy (TCR)', fontweight='bold', fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'fig5_auroc_vs_ppvnpv_mae_tcr')

# Plot BCR
if bcr_scatter_data:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    for model, ts, delta, ppv_mae, npv_mae in bcr_scatter_data:
        c = BCR_MODEL_COLORS[model]
        if not np.isnan(ppv_mae):
            ax.scatter(ppv_mae, delta, c=c, marker='o', s=30, alpha=0.7,
                       edgecolors='white', linewidth=0.3)
        if not np.isnan(npv_mae):
            ax.scatter(npv_mae, delta, c=c, marker='s', s=30, alpha=0.7,
                       edgecolors='white', linewidth=0.3)

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='PPV MAE'),
               Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=6, label='NPV MAE')]
    for m in BCR_MODELS:
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=BCR_MODEL_COLORS[m],
                               markersize=5, label=BCR_DISPLAY[m]))
    ax.legend(handles=handles, fontsize=5, loc='lower right', framealpha=0.8, ncol=2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('PPV/NPV prediction MAE', fontsize=8)
    ax.set_ylabel('ΔAUROC', fontsize=8)
    ax.set_title('Recalibration gain vs\nPPV/NPV accuracy (BCR)', fontweight='bold', fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'fig5_auroc_vs_ppvnpv_mae_bcr')


print("\nDone.")

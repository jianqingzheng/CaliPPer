#!/usr/bin/env python3
"""Fig 4 panels A-F with unique-averaged epitope/variant.

Each unique epitope/variant = one dot (averaged across folds/sources).
Suffix '_unique' on all output filenames for comparison with per-fold versions.

A_unique/B_unique: BLOSUM-RF epitope AP/AUROC (TCR, BLOSUM-sqrt)
C_unique/D_unique: RLEAAI antigen AP/AUROC (BCR, Lev)
E_unique: TCR epitope |error| scatter (model-averaged, unique)
F_unique: BCR variant |error| scatter (model-averaged, unique, SARS2/Influenza)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import hashlib

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
PANEL_DIR = os.path.join(FIG_DIR, 'fig4')
os.makedirs(PANEL_DIR, exist_ok=True)
from style_config import apply_publication_style
apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig3_fig4_tcr_cache')
BCR_CACHE = os.path.join(RESULTS, 'fig3_fig4_bcr_cache')
BCR_MAPPING_PATH = os.path.join(INPUT_DIR, 'fig4_meta', 'bcr_variant_name_mapping.csv')

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
TCR_COLOR = '#1f77b4'
BCR_COLOR = '#ff7f0e'


def save(fig, name):
    for dist_dir in ['blosum-sqrt', 'lev-logtransf']:
        out_dir = os.path.join(PANEL_DIR, dist_dir)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name + '.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, name + '.png'), dpi=200, bbox_inches='tight')
    # Also save to BCR_panels for panel_path resolution
    for dist_dir in ['blosum-sqrt/BCR_panels', 'lev-logtransf/BCR_panels']:
        out_dir = os.path.join(PANEL_DIR, dist_dir)
        os.makedirs(out_dir, exist_ok=True)
        if 'bcr' in name:
            fig.savefig(os.path.join(out_dir, name + '.png'), dpi=200, bbox_inches='tight')
    # TCR_panels
    for dist_dir in ['blosum-sqrt/TCR_panels']:
        out_dir = os.path.join(PANEL_DIR, dist_dir)
        os.makedirs(out_dir, exist_ok=True)
        if 'bcr' not in name:
            fig.savefig(os.path.join(out_dir, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}')


def load_and_unique_average(cache_dir, prefix, split, strategy, metric, model=None):
    """Load cached predictions and average per unique subset."""
    f = os.path.join(cache_dir, f'{prefix}_{split}_{strategy}_{metric}.csv')
    if not os.path.exists(f):
        print(f"  WARNING: cache file not found: {os.path.basename(f)}")
        return None
    df = pd.read_csv(f)
    if model:
        df = df[df['model'] == model]

    # For CT data, 'subset' may be 'variant' (placeholder) — use 'source' as key
    if 'source' in df.columns and df['subset'].nunique() == 1:
        group_col = 'source'
    else:
        group_col = 'subset'

    avg = df.groupby(group_col).agg(
        predicted=('predicted', 'mean'),
        actual=('actual', 'mean'),
        n=('n', 'sum'),           # total samples across all folds
        prevalence=('prevalence', 'mean'),
    ).reset_index()
    avg.rename(columns={group_col: 'subset'}, inplace=True)
    return avg


def plot_property_scatter(ax, avg, metric_disp, title):
    """Property-style scatter: size=n (total), color=prevalence, with r/MAE annotation."""
    # Log-scale sizing: visible range from small to large subsets
    n_vals = avg['n'].values.astype(float)
    n_min, n_max = n_vals.min(), max(n_vals.max(), n_vals.min() + 1)
    sizes = 30 + 250 * (np.log1p(n_vals) - np.log1p(n_min)) / (np.log1p(n_max) - np.log1p(n_min))

    sc = ax.scatter(avg['actual'], avg['predicted'], s=sizes, c=avg['prevalence'],
                     cmap='coolwarm', vmin=0, vmax=1, alpha=0.7,
                     edgecolors='white', linewidth=0.5)

    # Diagonal
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8, alpha=0.5)

    # Stats
    r, p = pearsonr(avg['actual'], avg['predicted'])
    mae = (avg['predicted'] - avg['actual']).abs().mean()
    p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    ax.text(0.05, 0.95, f'r = {r:.2f}, {p_str}\nMAE = {mae:.3f}, n = {len(avg)}',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'Actual {metric_disp}', fontsize=8)
    ax.set_ylabel(f'Predicted {metric_disp}', fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)  # grid behind data

    # Size legend with representative n values
    n_examples = [int(n_min), int(np.median(n_vals)), int(n_max)]
    # Deduplicate and format
    seen = set()
    for n_ex in n_examples:
        if n_ex in seen: continue
        seen.add(n_ex)
        s = 30 + 250 * (np.log1p(n_ex) - np.log1p(n_min)) / (np.log1p(n_max) - np.log1p(n_min))
        label = f'n={n_ex:,}' if n_ex < 10000 else f'n={n_ex/1000:.1f}k'
        ax.scatter([], [], s=s, c='gray', alpha=0.5, label=label)
    ax.legend(fontsize=5.5, loc='lower right', framealpha=0.8,
              handletextpad=0.3, labelspacing=0.5)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Class prevalence', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    return r, mae


# ═══════════════════════════════════════════
# Panels A_unique/B_unique: BLOSUM-RF epitope (TCR)
# ═══════════════════════════════════════════
print("=== Panels A/B unique: BLOSUM-RF epitope ===")

for metric, metric_disp in [('ap', 'AP'), ('aucroc', 'AUROC')]:
    parts = []
    missing_splits = []
    for split in ['cv', 'ct']:
        avg = load_and_unique_average(TCR_CACHE, 'tcr_fig4_blosum-sqrt', split, 'epitope', metric, model='blosum_rf')
        if avg is not None:
            parts.append(avg)
        else:
            missing_splits.append(split)
    if missing_splits:
        print(f"  ⚠ {metric_disp}: missing splits {missing_splits} — panel will be INCOMPLETE (CV-only)")
    if not parts:
        continue
    combined = pd.concat(parts, ignore_index=True)
    # Average again in case same epitope appears in both CV and CT
    final = combined.groupby('subset').agg(
        predicted=('predicted', 'mean'), actual=('actual', 'mean'),
        n=('n', 'mean'), prevalence=('prevalence', 'mean'),
    ).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    r, mae = plot_property_scatter(ax, final, metric_disp,
                                     f'BLOSUM-RF: epitope splitting, {metric_disp}\n(unique-averaged)')
    save(fig, f'fig4_property_epitope_{metric}_blosum_rf_unique')
    print(f"  {metric_disp}: n={len(final)}, r={r:.3f}, MAE={mae:.4f}")

# ═══════════════════════════════════════════
# Panels C_unique/D_unique: RLEAAI antigen (BCR)
# ═══════════════════════════════════════════
print("\n=== Panels C/D unique: RLEAAI antigen ===")

for metric, metric_disp in [('ap', 'AP'), ('aucroc', 'AUROC')]:
    parts = []
    missing_splits = []
    for split in ['cv', 'ct']:
        avg = load_and_unique_average(BCR_CACHE, 'bcr_fig4_fold4cal', split, 'antigen', metric, model='rleaai')
        if avg is not None:
            parts.append(avg)
        else:
            missing_splits.append(split)
    if missing_splits:
        print(f"  ⚠ {metric_disp}: missing splits {missing_splits} — panel will be INCOMPLETE")
    if not parts:
        continue
    combined = pd.concat(parts, ignore_index=True)
    final = combined.groupby('subset').agg(
        predicted=('predicted', 'mean'), actual=('actual', 'mean'),
        n=('n', 'mean'), prevalence=('prevalence', 'mean'),
    ).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    r, mae = plot_property_scatter(ax, final, metric_disp,
                                     f'RLEAAI: antigen splitting, {metric_disp}\n(unique-averaged)')
    save(fig, f'fig4_bcr_property_antigen_{metric}_rleaai_unique')
    print(f"  {metric_disp}: n={len(final)}, r={r:.3f}, MAE={mae:.4f}")

# ═══════════════════════════════════════════
# Panel E_unique: TCR epitope |error| scatter (model-averaged, unique)
# ═══════════════════════════════════════════
print("\n=== Panel E unique: TCR epitope |error| scatter ===")

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
handles = []

for metric, marker, label in [('ap', 'o', 'AP'), ('aucroc', 's', 'AUROC')]:
    parts = []
    for split in ['cv', 'ct']:
        df_path = os.path.join(TCR_CACHE, f'tcr_fig4_blosum-sqrt_{split}_epitope_{metric}.csv')
        if os.path.exists(df_path):
            parts.append(pd.read_csv(df_path))
    if not parts: continue
    df = pd.concat(parts, ignore_index=True)
    # Average predicted and actual across models AND folds per unique epitope FIRST,
    # then compute |error| (not the other way — Jensen's inequality)
    avg = df.groupby('subset').agg(
        predicted=('predicted', 'mean'), actual=('actual', 'mean'), n=('n', 'sum'),
    ).reset_index()
    avg['abs_error'] = (avg['predicted'] - avg['actual']).abs()

    n_vals = avg['n'].values.astype(float)
    n_min, n_max = n_vals.min(), max(n_vals.max(), n_vals.min() + 1)
    sizes = 30 + 250 * (np.log1p(n_vals) - np.log1p(n_min)) / (np.log1p(n_max) - np.log1p(n_min))
    ax.scatter(avg['actual'], avg['abs_error'], marker=marker, s=sizes, alpha=0.5,
               c=TCR_COLOR, edgecolors='white', linewidth=0.5)

    valid = ~(np.isnan(avg['actual']) | np.isnan(avg['abs_error']))
    if valid.sum() >= 4:
        r, p = pearsonr(avg['actual'][valid], avg['abs_error'][valid])
        p_str = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
        handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=TCR_COLOR,
                               markersize=6, label=f'{label} (r={r:.2f}, {p_str})'))
        # Correlation line
        x_fit = np.linspace(avg['actual'].min(), avg['actual'].max(), 50)
        slope = r * avg['abs_error'].std() / avg['actual'].std()
        intercept = avg['abs_error'].mean() - slope * avg['actual'].mean()
        ax.plot(x_fit, slope * x_fit + intercept, '--', color=TCR_COLOR, alpha=0.5, linewidth=1)
        print(f"  {metric}: n={len(avg)}, r={r:.3f}")

ax.set_xlabel('Actual metric value', fontsize=8)
ax.set_ylabel('|Prediction error|', fontsize=8)
ax.set_title('TCR epitope prediction error\n(unique-averaged)', fontweight='bold', fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.35)
ax.grid(True, alpha=0.3); ax.set_axisbelow(True)  # grid behind data
# Size legend
for n_ex, label in [(200, 'n=200'), (2000, 'n=2k'), (5000, 'n=5k')]:
    s = 30 + 250 * (np.log1p(n_ex) - np.log1p(n_min)) / (np.log1p(n_max) - np.log1p(n_min))
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markersize=np.sqrt(s), label=label, alpha=0.5))
ax.legend(handles=handles, fontsize=5.5, loc='upper right', framealpha=0.8,
          handletextpad=0.3, labelspacing=0.4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'fig4_tcr_epitope_error_scatter_unique')

# ═══════════════════════════════════════════
# Panel F_unique: BCR variant |error| scatter (model-averaged, unique)
# ═══════════════════════════════════════════
print("\n=== Panel F unique: BCR variant |error| scatter ===")

bcr_mapping = pd.read_csv(BCR_MAPPING_PATH) if os.path.exists(BCR_MAPPING_PATH) else None
hash_to_pathogen = {}
if bcr_mapping is not None:
    for _, row in bcr_mapping.iterrows():
        vhash = hashlib.md5(row['variant_seq'].encode()).hexdigest()[:12]
        hash_to_pathogen[vhash] = 'Influenza' if row['data_source'] == 'flu' else 'SARS2'

parts = []
for split in ['cv', 'ct']:
    df_path = os.path.join(BCR_CACHE, f'bcr_fig4_fold4cal_{split}_antigen_ap.csv')
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        # For CT, use source as subset key
        if 'source' in df.columns and df['subset'].nunique() == 1:
            df['subset'] = df['source']
        parts.append(df)

if parts:
    df = pd.concat(parts, ignore_index=True)
    # Average predicted and actual first, then compute |error|
    avg = df.groupby('subset').agg(
        predicted=('predicted', 'mean'), actual=('actual', 'mean'), n=('n', 'sum'),
    ).reset_index()
    avg['abs_error'] = (avg['predicted'] - avg['actual']).abs()

    # Pathogen mapping: CV keys are hashes, CT keys have 'sars_'/'flu_' prefix
    def resolve_pathogen(key):
        if key in hash_to_pathogen:
            return hash_to_pathogen[key]
        # Strip prefix for CT keys like 'sars_2c5dba60a448'
        if '_' in key:
            prefix = key.split('_')[0]
            if prefix == 'flu':
                return 'Influenza'
            return 'SARS2'
        return 'SARS2'
    avg['pathogen'] = avg['subset'].apply(resolve_pathogen)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    handles = []
    all_actual, all_error = [], []

    for pathogen, marker, color, plabel in [('SARS2', 'o', '#3498db', 'SARS-CoV-2'),
                                             ('Influenza', '^', '#2ecc71', 'Influenza')]:
        sub = avg[avg['pathogen'] == pathogen]
        if len(sub) > 0:
            n_vals_sub = sub['n'].values.astype(float)
            n_min_all, n_max_all = avg['n'].min(), max(avg['n'].max(), avg['n'].min() + 1)
            sizes = 30 + 250 * (np.log1p(n_vals_sub) - np.log1p(n_min_all)) / (np.log1p(n_max_all) - np.log1p(n_min_all))
            ax.scatter(sub['actual'], sub['abs_error'], marker=marker, s=sizes, alpha=0.5,
                       c=color, edgecolors='white', linewidth=0.5)
            handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                                   markersize=6, label=plabel))
            all_actual.extend(sub['actual']); all_error.extend(sub['abs_error'])

    all_actual, all_error = np.array(all_actual), np.array(all_error)
    if len(all_actual) >= 4:
        r, p = pearsonr(all_actual, all_error)
        p_str = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
        x_fit = np.linspace(all_actual.min(), all_actual.max(), 50)
        slope = r * all_error.std() / all_actual.std()
        intercept = all_error.mean() - slope * all_actual.mean()
        ax.plot(x_fit, slope * x_fit + intercept, '--', color='gray', alpha=0.6, linewidth=1)
        ax.text(0.03, 0.97, f'r={r:.2f}, {p_str}\nn={len(avg)}',
                transform=ax.transAxes, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        print(f"  n={len(avg)}, r={r:.3f}")

    ax.set_xlabel('Actual AP', fontsize=8)
    ax.set_ylabel('|Prediction error|', fontsize=8)
    ax.set_title('BCR variant prediction error\n(unique-averaged, AP)', fontweight='bold', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.35)
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)  # grid behind data
    # Size legend
    n_min_all, n_max_all = avg['n'].min(), avg['n'].max()
    for n_ex, label in [(50, 'n=50'), (500, 'n=500'), (2000, 'n=2k')]:
        s = 30 + 250 * (np.log1p(n_ex) - np.log1p(n_min_all)) / (np.log1p(n_max_all) - np.log1p(n_min_all))
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                               markersize=np.sqrt(s), label=label, alpha=0.5))
    ax.legend(handles=handles, fontsize=5.5, loc='upper right', framealpha=0.8,
              handletextpad=0.3, labelspacing=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'fig4_bcr_variant_error_scatter_unique')

# ═══════════════════════════════════════════
# Summary comparison
# ═══════════════════════════════════════════
print("\n=== SUMMARY: per-fold vs unique-averaged ===")
print(f"{'Panel':10s} | {'Per-fold n':>12s} {'r':>8s} | {'Unique n':>12s} {'r':>8s}")
print("-" * 60)

for model, prefix, cache, strategy in [
    ('blosum_rf', 'tcr_fig4_blosum-sqrt', TCR_CACHE, 'epitope'),
    ('rleaai', 'bcr_fig4_fold4cal', BCR_CACHE, 'antigen'),
]:
    for metric in ['ap', 'aucroc']:
        parts_pf, parts_ua = [], []
        for split in ['cv', 'ct']:
            f = os.path.join(cache, f'{prefix}_{split}_{strategy}_{metric}.csv')
            if os.path.exists(f):
                df = pd.read_csv(f)
                df_m = df[df['model'] == model]
                parts_pf.append(df_m)
                # unique
                if 'source' in df_m.columns and df_m['subset'].nunique() == 1:
                    avg = df_m.groupby('source').agg(predicted=('predicted','mean'), actual=('actual','mean')).reset_index()
                else:
                    avg = df_m.groupby('subset').agg(predicted=('predicted','mean'), actual=('actual','mean')).reset_index()
                parts_ua.append(avg)

        if parts_pf:
            pf = pd.concat(parts_pf)
            r_pf, _ = pearsonr(pf['predicted'], pf['actual'])
            ua = pd.concat(parts_ua)
            ua2 = ua.groupby(ua.columns[0]).agg(predicted=('predicted','mean'), actual=('actual','mean')).reset_index()
            r_ua, _ = pearsonr(ua2['predicted'], ua2['actual'])
            domain = 'TCR' if 'tcr' in prefix else 'BCR'
            print(f"{domain} {metric:6s} | n={len(pf):4d}     r={r_pf:.3f} | n={len(ua2):4d}     r={r_ua:.3f}")

print("\nDone.")
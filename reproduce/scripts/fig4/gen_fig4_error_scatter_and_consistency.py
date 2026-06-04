#!/usr/bin/env python3
"""Fig 4 panels E-H:
  E: TCR epitope |error| vs actual (AP=circle, AUROC=square), model-averaged, CV+CT
     marker size = subset sample count, dashed correlation line with r/p
  F: BCR variant |error| vs actual AP, model-averaged, CV+CT
     shape = SARS2 (circle) / Influenza (triangle), marker size = subset count
  G: MAE horizontal boxplot across 10 models (AP), same format as fig2 panel O
  H: Correlation horizontal boxplot across 10 models (AP)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

TCR_MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
TCR_DISPLAY = {'nettcr': 'NetTCR', 'atm_tcr': 'ATM-TCR', 'blosum_rf': 'BLOSUM-RF',
               'ergo_ii': 'ERGO-II', 'tcrbert': 'TCR-BERT'}
BCR_DISPLAY = {'xbcr': 'XBCR-net', 'deepaai': 'DeepAAI', 'mambaaai': 'MambaAAI',
               'mint': 'MINT', 'rleaai': 'RLEAAI'}
TCR_COLOR = '#1f77b4'
BCR_COLOR = '#ff7f0e'

TCR_DIST = 'blosum-sqrt'
BCR_DIST = 'fold4cal'

BCR_MAPPING_PATH = os.path.join(INPUT_DIR, 'fig4_meta', 'bcr_variant_name_mapping.csv')


def save(fig, name):
    for dist_dir in ['blosum-sqrt', 'lev-logtransf']:
        out_dir = os.path.join(PANEL_DIR, dist_dir)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name + '.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}')


def load_tcr(split, strategy, metric):
    f = os.path.join(TCR_CACHE, f'tcr_fig4_{TCR_DIST}_{split}_{strategy}_{metric}.csv')
    return pd.read_csv(f) if os.path.exists(f) else None


def load_bcr(split, strategy, metric):
    f = os.path.join(BCR_CACHE, f'bcr_fig4_{BCR_DIST}_{split}_{strategy}_{metric}.csv')
    return pd.read_csv(f) if os.path.exists(f) else None


# ═══════════════════════════════════════════
# Panel E: TCR epitope |error| scatter (AP + AUROC)
# ═══════════════════════════════════════════
print("=== Panel E: TCR epitope |error| scatter ===")

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
handles = []

for metric, marker, label in [('ap', 'o', 'AP'), ('aucroc', 's', 'AUROC')]:
    parts = []
    for split in ['cv', 'ct']:
        df = load_tcr(split, 'epitope', metric)
        if df is not None:
            parts.append(df)
    if not parts:
        continue
    df = pd.concat(parts, ignore_index=True)
    df['abs_error'] = (df['predicted'] - df['actual']).abs()

    # Average across models per (subset, fold/source)
    # Each unique (subset, fold) = one dot, averaged over 5 models
    if 'fold' in df.columns:
        avg = df.groupby(['subset', 'fold']).agg(
            actual=('actual', 'mean'),
            abs_error=('abs_error', 'mean'),
            n=('n', 'first'),
        ).reset_index()
    else:
        avg = df.groupby('subset').agg(
            actual=('actual', 'mean'),
            abs_error=('abs_error', 'mean'),
            n=('n', 'first'),
        ).reset_index()

    # Size by sample count (log-scale, large range for visibility)
    n_vals = avg['n'].values.astype(float)
    sizes = 20 + 200 * (np.log1p(n_vals) - np.log1p(n_vals.min())) / max(np.log1p(n_vals.max()) - np.log1p(n_vals.min()), 1)

    ax.scatter(avg['actual'], avg['abs_error'], marker=marker, s=sizes, alpha=0.5,
               c=TCR_COLOR, edgecolors='white', linewidth=0.5)

    # Correlation line
    valid = ~(np.isnan(avg['actual']) | np.isnan(avg['abs_error']))
    if valid.sum() >= 4:
        r, p = pearsonr(avg['actual'][valid], avg['abs_error'][valid])
        x_fit = np.linspace(avg['actual'].min(), avg['actual'].max(), 50)
        slope = r * avg['abs_error'].std() / avg['actual'].std()
        intercept = avg['abs_error'].mean() - slope * avg['actual'].mean()
        ax.plot(x_fit, slope * x_fit + intercept, '--', color=TCR_COLOR, alpha=0.5, linewidth=1)
        p_str = f'p<0.001' if p < 0.001 else f'p={p:.3f}'
        handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=TCR_COLOR,
                               markersize=6, label=f'{label} (r={r:.2f}, {p_str})'))
        print(f"  {metric}: n={len(avg)}, r={r:.3f}, p={p:.4f}")

ax.set_xlabel('Actual metric value', fontsize=8)
ax.set_ylabel('|Prediction error|', fontsize=8)
ax.set_title('TCR epitope prediction error\n(model-averaged, CV+CT)', fontweight='bold', fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.35)
# Size legend
for n_example, label in [(150, 'n=150'), (500, 'n=500'), (1500, 'n=1.5k')]:
    s = 20 + 200 * (np.log1p(n_example) - np.log1p(129)) / max(np.log1p(1509) - np.log1p(129), 1)
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markersize=np.sqrt(s), label=label, alpha=0.5))
ax.legend(handles=handles, fontsize=5.5, loc='upper right', framealpha=0.8,
          handletextpad=0.3, labelspacing=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save(fig, 'fig4_tcr_epitope_error_scatter')

# ═══════════════════════════════════════════
# Panel F: BCR variant |error| scatter (AP, SARS2/Influenza)
# ═══════════════════════════════════════════
print("\n=== Panel F: BCR variant |error| scatter ===")

# Load variant → pathogen mapping
bcr_mapping = pd.read_csv(BCR_MAPPING_PATH) if os.path.exists(BCR_MAPPING_PATH) else None
hash_to_pathogen = {}
if bcr_mapping is not None:
    for _, row in bcr_mapping.iterrows():
        vhash = hashlib.md5(row['variant_seq'].encode()).hexdigest()[:12]
        pathogen = 'Influenza' if row['data_source'] == 'flu' else 'SARS2'
        hash_to_pathogen[vhash] = pathogen

parts = []
for split in ['cv', 'ct']:
    df = load_bcr(split, 'antigen', 'ap')
    if df is not None:
        parts.append(df)
if parts:
    df = pd.concat(parts, ignore_index=True)
    df['abs_error'] = (df['predicted'] - df['actual']).abs()

    # Average across models per (subset[, fold]). The audit-derived BCR cache
    # CSVs do not include a 'fold' column (only subset/metric/predicted/actual/
    # ...), so we group by subset alone when fold is absent.
    groupby_cols = ['subset', 'fold'] if 'fold' in df.columns else ['subset']
    avg = df.groupby(groupby_cols).agg(
        actual=('actual', 'mean'),
        abs_error=('abs_error', 'mean'),
        n=('n', 'first'),
    ).reset_index()
    avg['pathogen'] = avg['subset'].map(hash_to_pathogen).fillna('SARS2')

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    handles = []
    all_actual, all_error = [], []

    for pathogen, marker, color, label in [('SARS2', 'o', '#3498db', 'SARS-CoV-2'),
                                            ('Influenza', '^', '#2ecc71', 'Influenza')]:
        sub = avg[avg['pathogen'] == pathogen]
        if len(sub) > 0:
            n_vals = sub['n'].values.astype(float)
            all_n = avg['n'].values.astype(float)
            sizes = 20 + 200 * (np.log1p(n_vals) - np.log1p(all_n.min())) / max(np.log1p(all_n.max()) - np.log1p(all_n.min()), 1)
            ax.scatter(sub['actual'], sub['abs_error'], marker=marker, s=sizes, alpha=0.5,
                       c=color, edgecolors='white', linewidth=0.5)
            handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                                   markersize=6, label=label))
            all_actual.extend(sub['actual'].tolist())
            all_error.extend(sub['abs_error'].tolist())

    # Overall correlation line
    all_actual, all_error = np.array(all_actual), np.array(all_error)
    valid = ~(np.isnan(all_actual) | np.isnan(all_error))
    if valid.sum() >= 4:
        r, p = pearsonr(all_actual[valid], all_error[valid])
        x_fit = np.linspace(all_actual.min(), all_actual.max(), 50)
        slope = r * all_error.std() / all_actual.std()
        intercept = all_error.mean() - slope * all_actual.mean()
        ax.plot(x_fit, slope * x_fit + intercept, '--', color='gray', alpha=0.6, linewidth=1)
        p_str = f'p<0.001' if p < 0.001 else f'p={p:.3f}'
        ax.text(0.03, 0.97, f'r={r:.2f}, {p_str}\nn={len(avg)}',
                transform=ax.transAxes, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        print(f"  n={len(avg)}, r={r:.3f}, p={p:.4f}")

    ax.set_xlabel('Actual AP', fontsize=8)
    ax.set_ylabel('|Prediction error|', fontsize=8)
    ax.set_title('BCR variant prediction error\n(model-averaged, AP)', fontweight='bold', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.35)
    # Size legend
    n_min, n_max = avg['n'].min(), avg['n'].max()
    for n_example, label in [(30, 'n=30'), (100, 'n=100'), (400, 'n=400')]:
        s = 20 + 200 * (np.log1p(n_example) - np.log1p(n_min)) / max(np.log1p(n_max) - np.log1p(n_min), 1)
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                               markersize=np.sqrt(s), label=label, alpha=0.5))
    ax.legend(handles=handles, fontsize=5.5, loc='upper right', framealpha=0.8,
              handletextpad=0.3, labelspacing=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, 'fig4_bcr_variant_error_scatter')

# ═══════════════════════════════════════════
# Panel G: MAE horizontal boxplot across 10 models (AP)
# ═══════════════════════════════════════════
print("\n=== Panel G: MAE boxplot (AP) ===")

model_maes = {}

def get_group_col(df):
    """Return 'fold' for CV data, 'source' for CT data."""
    return 'fold' if 'fold' in df.columns else 'source'

# TCR: per-model per-fold/source MAE
for split in ['cv', 'ct']:
    df = load_tcr(split, 'epitope', 'ap')
    if df is None:
        continue
    df['abs_error'] = (df['predicted'] - df['actual']).abs()
    gcol = get_group_col(df)
    for model in TCR_MODELS:
        sub = df[df['model'] == model]
        if len(sub) > 0:
            if model not in model_maes:
                model_maes[model] = {'per_fold': [], 'type': 'TCR'}
            for grp in sub[gcol].unique():
                fsub = sub[sub[gcol] == grp]
                model_maes[model]['per_fold'].append(fsub['abs_error'].mean())

# BCR
for split in ['cv', 'ct']:
    df = load_bcr(split, 'antigen', 'ap')
    if df is None:
        continue
    df['abs_error'] = (df['predicted'] - df['actual']).abs()
    gcol = get_group_col(df)
    for model in BCR_MODELS:
        sub = df[df['model'] == model]
        if len(sub) > 0:
            if model not in model_maes:
                model_maes[model] = {'per_fold': [], 'type': 'BCR'}
            for grp in sub[gcol].unique():
                fsub = sub[sub[gcol] == grp]
                model_maes[model]['per_fold'].append(fsub['abs_error'].mean())

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
all_labels, all_data, all_colors = [], [], []
y_positions = []
y_idx = 0

for model in TCR_MODELS:
    if model in model_maes:
        all_labels.append(TCR_DISPLAY[model])
        all_data.append(model_maes[model]['per_fold'])
        all_colors.append(TCR_COLOR)
        y_positions.append(y_idx)
        y_idx += 1

y_idx += 0.5

for model in BCR_MODELS:
    if model in model_maes:
        all_labels.append(BCR_DISPLAY[model])
        all_data.append(model_maes[model]['per_fold'])
        all_colors.append(BCR_COLOR)
        y_positions.append(y_idx)
        y_idx += 1

for i, (pos, data, color) in enumerate(zip(y_positions, all_data, all_colors)):
    bp = ax.boxplot([data], positions=[pos], widths=0.6, vert=False,
                     patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.5)
    bp['medians'][0].set_color('black')
    bp['medians'][0].set_linewidth(1.2)
    jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(data))
    ax.scatter(data, np.full(len(data), pos) + jitter,
               c=color, s=15, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

ax.set_yticks(y_positions)
ax.set_yticklabels(all_labels, fontsize=6)
ax.set_xlabel('MAE (AP subset prediction)', fontsize=8)
ax.set_title('Subset prediction MAE\n(CV+CT, AP)', fontweight='bold', fontsize=9)
ax.set_xlim(0.0, 0.35)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend([Patch(facecolor=TCR_COLOR, alpha=0.5),
           Patch(facecolor=BCR_COLOR, alpha=0.5)],
          ['TCR', 'BCR'], fontsize=7, loc='upper right', framealpha=0.8)
save(fig, 'fig4_subset_mae_boxplot')

# ═══════════════════════════════════════════
# Panel H: Correlation horizontal boxplot across 10 models (AP)
# ═══════════════════════════════════════════
print("\n=== Panel H: Correlation boxplot (AP) ===")

model_corrs = {}

# TCR
for split in ['cv', 'ct']:
    df = load_tcr(split, 'epitope', 'ap')
    if df is None:
        continue
    gcol = get_group_col(df)
    for model in TCR_MODELS:
        sub = df[df['model'] == model]
        if model not in model_corrs:
            model_corrs[model] = {'rs': [], 'type': 'TCR'}
        for grp in sub[gcol].unique():
            fsub = sub[sub[gcol] == grp]
            if len(fsub) >= 4:
                r, _ = pearsonr(fsub['predicted'], fsub['actual'])
                model_corrs[model]['rs'].append(abs(r))

# BCR
for split in ['cv', 'ct']:
    df = load_bcr(split, 'antigen', 'ap')
    if df is None:
        continue
    gcol = get_group_col(df)
    for model in BCR_MODELS:
        sub = df[df['model'] == model]
        if model not in model_corrs:
            model_corrs[model] = {'rs': [], 'type': 'BCR'}
        for grp in sub[gcol].unique():
            fsub = sub[sub[gcol] == grp]
            if len(fsub) >= 4:
                r, _ = pearsonr(fsub['predicted'], fsub['actual'])
                model_corrs[model]['rs'].append(abs(r))

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
all_labels, all_data, all_colors = [], [], []
y_positions = []
y_idx = 0

for model in TCR_MODELS:
    if model in model_corrs and model_corrs[model]['rs']:
        all_labels.append(TCR_DISPLAY[model])
        all_data.append(model_corrs[model]['rs'])
        all_colors.append(TCR_COLOR)
        y_positions.append(y_idx)
        y_idx += 1

y_idx += 0.5

for model in BCR_MODELS:
    if model in model_corrs and model_corrs[model]['rs']:
        all_labels.append(BCR_DISPLAY[model])
        all_data.append(model_corrs[model]['rs'])
        all_colors.append(BCR_COLOR)
        y_positions.append(y_idx)
        y_idx += 1

for i, (pos, data, color) in enumerate(zip(y_positions, all_data, all_colors)):
    bp = ax.boxplot([data], positions=[pos], widths=0.6, vert=False,
                     patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.5)
    bp['medians'][0].set_color('black')
    bp['medians'][0].set_linewidth(1.2)
    jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(data))
    ax.scatter(data, np.full(len(data), pos) + jitter,
               c=color, s=15, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

ax.set_yticks(y_positions)
ax.set_yticklabels(all_labels, fontsize=6)
ax.set_xlabel('|Pearson r| (AP subset prediction)', fontsize=8)
ax.set_title('Subset prediction correlation\n(CV+CT, AP)', fontweight='bold', fontsize=9)
ax.set_xlim(0.0, 1.05)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend([Patch(facecolor=TCR_COLOR, alpha=0.5),
           Patch(facecolor=BCR_COLOR, alpha=0.5)],
          ['TCR', 'BCR'], fontsize=7, loc='upper left', framealpha=0.8)
save(fig, 'fig4_subset_correlation_boxplot')

print("\nDone.")

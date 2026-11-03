#!/usr/bin/env python3
"""Fair comparison of distance metrics for deepAntigen degradation analysis.

All methods use the same data (Dataset 8, 1,714 zero-shot pairs) and same
8-equal-count binning. Only the distance function differs.

Methods compared:
  1. SW-max (baseline):     BLOSUM62 SW similarity, max over 208 unique training peptides
  2. S2DD-Lev (sigma_C):    Levenshtein LogDist, topK=50 mean, sigma_C weights (99.97% peptide)
  3. S2DD-Lev (uniform):    Levenshtein LogDist, topK=50 mean, 50/50 weights
  4. S2DD-SW-peptide:       SW similarity on peptide chain only (mimics SW-max but via S2DD pipeline)

None of these methods uses labels or model predictions — all are unsupervised
sequence distance metrics computed only from training vs test sequences.
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import parasail

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
# CaliPPer self-contained path bootstrap: writes distances into INPUT_DIR/results/
# so Stage 1 (compute_fig6_recal_data.py) consumes from-scratch values.
sys.path.insert(0, str(SCRIPT_DIR.parent))   # /.../reproduce/scripts/  (for style_config + _paths)
sys.path.insert(0, str(SCRIPT_DIR))           # /.../reproduce/scripts/fig6/
from _paths import INPUT_DIR                  # _paths.py also adds CaliPPer/ to sys.path
PROJECT_ROOT = Path(INPUT_DIR)

from style_config import apply_publication_style, DPI, COL_DOUBLE, FONT_LABEL, FONT_TICK
from calipper.general_evaluator import safe_metric
from calipper.combine_first_helpers import compute_combine_first_distances

apply_publication_style()

# CaliPPer paths: results go into INPUT_DIR/results/ so Stage 1 reads fresh values.
OUTPUT_DIR_PLOTS = SCRIPT_DIR.parent.parent / 'figures' / 'output' / 'fig6'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'deepantigen_retrospective'
DATA_DIR = PROJECT_ROOT / 'Data' / 'tcr_seq' / 'proc_files' / 'deepantigen_data'

N_BINS = 8
METRICS = ['aucroc', 'ap', 'f1']
METRIC_LABELS = {'aucroc': 'AUROC', 'ap': 'AP', 'f1': 'F1'}
CACHE_SW = RESULTS_DIR / 's2dd_degradation' / 'zero_shot_sw_distances.csv'
CACHE_UNIFORM = RESULTS_DIR / 's2dd_degradation' / 'zero_shot_uniform_distances.csv'
CACHE_SW_TOPK = RESULTS_DIR / 's2dd_degradation' / 'zero_shot_sw_topk_distances.csv'


def compute_sw_similarity_max(test_peptides, train_peptides):
    """MAX normalized SW similarity to any training peptide."""
    train_self = {s: parasail.sw_stats(s, s, 10, 1, parasail.blosum62).score for s in train_peptides}
    max_sims = []
    t0 = time.time()
    for i, q in enumerate(test_peptides):
        q_self = parasail.sw_stats(q, q, 10, 1, parasail.blosum62).score
        best = 0.0
        for r in train_peptides:
            s = parasail.sw_stats(q, r, 10, 1, parasail.blosum62).score
            denom = np.sqrt(q_self * train_self[r])
            if denom > 0:
                best = max(best, s / denom)
        max_sims.append(best)
        if (i + 1) % 200 == 0 or i == len(test_peptides) - 1:
            print(f"    [{i+1:>5}/{len(test_peptides)}] {(i+1)/len(test_peptides)*100:.1f}%")
    print(f"  Done in {time.time()-t0:.1f}s")
    return np.array(max_sims)


def compute_sw_topk_distance(test_peptides, train_peptides_rows, K=50):
    """S2DD-style: top-K SW similarity (over all training rows), convert to distance.

    Mimics S2DD's topK aggregation: for each test peptide, compute SW similarity
    to all training peptides (as they appear in training rows, with frequency),
    take top-K highest similarities, average them, convert to distance.
    """
    unique_train_peps = np.unique(train_peptides_rows)
    train_self = {s: parasail.sw_stats(s, s, 10, 1, parasail.blosum62).score for s in unique_train_peps}

    # Compute sim(test_pep, unique_train_pep) for all pairs
    print(f"  Computing SW similarity matrix: {len(test_peptides)} × {len(unique_train_peps)}")
    sim_matrix = {}
    t0 = time.time()
    for i, q in enumerate(test_peptides):
        if q not in sim_matrix:
            q_self = parasail.sw_stats(q, q, 10, 1, parasail.blosum62).score
            sims = []
            for r in unique_train_peps:
                s = parasail.sw_stats(q, r, 10, 1, parasail.blosum62).score
                denom = np.sqrt(q_self * train_self[r])
                sims.append(s / denom if denom > 0 else 0.0)
            sim_matrix[q] = np.array(sims)
        if (i + 1) % 200 == 0 or i == len(test_peptides) - 1:
            print(f"    [{i+1:>5}/{len(test_peptides)}] {(i+1)/len(test_peptides)*100:.1f}%")
    print(f"  SW matrix done in {time.time()-t0:.1f}s")

    # Index map: unique_pep → indices in train_peptides_rows
    pep_to_rows = {p: np.where(train_peptides_rows == p)[0] for p in unique_train_peps}

    # For each test peptide, expand to all training rows, take top-K
    distances = []
    for q in test_peptides:
        pep_sims = sim_matrix[q]  # 1 per unique training peptide
        # Expand: each training row has the sim corresponding to its peptide
        row_sims = np.zeros(len(train_peptides_rows))
        for pep_idx, p in enumerate(unique_train_peps):
            row_sims[pep_to_rows[p]] = pep_sims[pep_idx]
        # Top-K largest similarities = top-K smallest distances
        topk = np.sort(row_sims)[::-1][:K]  # top K most similar
        # Convert to distance: use log(1 - sim + eps)
        topk_dists = np.log(0.1 * (1.0 - topk + 0.1))
        distances.append(topk_dists.mean())
    return np.array(distances)


def bin_by_metric(distances, y_true, y_prob, n_bins=N_BINS):
    """Bin ascending (lower = more similar/closer), compute metrics per bin."""
    si = np.argsort(distances)
    bs = len(si) // n_bins
    out = {'mean_val': [], 'n': []}
    for m in METRICS:
        out[m] = []
    for i in range(n_bins):
        s = i * bs
        e = len(si) if i == n_bins - 1 else (i + 1) * bs
        idx = si[s:e]
        out['mean_val'].append(distances[idx].mean())
        out['n'].append(len(idx))
        for m in METRICS:
            out[m].append(safe_metric(m, y_true[idx], y_prob[idx]))
    return {k: np.array(v) for k, v in out.items()}


# ── Load data ────────────────────────────────────────────────────────────
print("Loading data...")
zs = pd.read_csv(RESULTS_DIR / 's2dd_degradation' / 'zero_shot_with_distances.csv')
train = pd.read_csv(DATA_DIR / 'train.csv')
train_peptides_rows = train['peptide'].values  # per-row (62,446)
train_peptides_unique = train['peptide'].unique()  # 208

y_true = zs['label'].values
y_prob = zs['prediction'].values

# Method 1: S2DD sigma_C (pre-computed — Levenshtein + sigma_C + weighted_max_znorm)
s2dd_sigma_dists = zs['s2dd_distance'].values

# Method 2: S2DD uniform (cached — Levenshtein + uniform + weighted_max_znorm)
if CACHE_UNIFORM.exists():
    s2dd_uniform_dists = pd.read_csv(CACHE_UNIFORM)['distance'].values
else:
    zs_r = zs.rename(columns={'binding_TCR': 'CDR3b'})
    train_r = train.rename(columns={'binding_TCR': 'CDR3b'})
    s2dd_uniform_dists = compute_combine_first_distances(
        zs_r, train_r, ['peptide', 'CDR3b'], np.array([0.5, 0.5]), 0.1, 0.1, 50)
    pd.DataFrame({'distance': s2dd_uniform_dists}).to_csv(CACHE_UNIFORM, index=False)

# Method 3: SW-max (baseline — BLOSUM + max over unique peptides)
if CACHE_SW.exists():
    sw_sims = pd.read_csv(CACHE_SW)['sw_similarity'].values
else:
    sw_sims = compute_sw_similarity_max(zs['peptide'].values, train_peptides_unique)
    pd.DataFrame({'sw_similarity': sw_sims}).to_csv(CACHE_SW, index=False)
sw_max_dists = 1.0 - sw_sims

# Method 4: SW-topK (BLOSUM + topK=50 mean over all training rows — S2DD-style aggregation with SW)
if CACHE_SW_TOPK.exists():
    sw_topk_dists = pd.read_csv(CACHE_SW_TOPK)['distance'].values
else:
    print("\nComputing SW-topK distances (BLOSUM-based S2DD-style aggregation)...")
    sw_topk_dists = compute_sw_topk_distance(zs['peptide'].values, train_peptides_rows, K=50)
    pd.DataFrame({'distance': sw_topk_dists}).to_csv(CACHE_SW_TOPK, index=False)
    print(f"Cached to {CACHE_SW_TOPK}")

# ── Bin and compute correlations ─────────────────────────────────────────
methods = [
    ('SW-max (BLOSUM + max)', sw_max_dists, '#95a5a6'),
    ('S2DD-Lev (sigma_C)', s2dd_sigma_dists, '#3498db'),
    ('S2DD-Lev (uniform)', s2dd_uniform_dists, '#2ecc71'),
    ('S2DD-SW (BLOSUM + topK)', sw_topk_dists, '#e67e22'),
]

results = {}
print("\n" + "="*90)
print("Distance metric comparison: per-bin degradation (Dataset 8, 8 equal bins)")
print("="*90)
print(f"{'Method':<30} {'Metric':<8} {'r':>8} {'p':>8} {'range':>8}")
print("-"*90)
for name, dists, _ in methods:
    bins = bin_by_metric(dists, y_true, y_prob)
    results[name] = bins
    for m in METRICS:
        r, p = pearsonr(bins['mean_val'], bins[m])
        rng = bins[m].max() - bins[m].min()
        print(f"{name:<30} {m:<8} {r:>+8.3f} {p:>8.3f} {rng:>8.4f}")

# ── Create figure: 4 rows × 3 cols ───────────────────────────────────────
print("\nCreating figure...")
fig, axes = plt.subplots(4, 3, figsize=(COL_DOUBLE * 2.0, COL_DOUBLE * 1.8), sharex=False)
panel_labels = ['a', 'b', 'c', 'd']
row_labels = ['SW-max\n(BLOSUM+max)', 'S2DD-Lev\n(sigma_C)', 'S2DD-Lev\n(uniform)', 'S2DD-SW\n(BLOSUM+topK)']

for row, (name, dists, color) in enumerate(methods):
    bins = results[name]
    for col, m in enumerate(METRICS):
        ax = axes[row, col]
        r, p = pearsonr(bins['mean_val'], bins[m])
        ax.scatter(bins['mean_val'], bins[m], s=50, c=color, alpha=0.8,
                   edgecolors='white', linewidth=0.6, zorder=3)
        z = np.polyfit(bins['mean_val'], bins[m], 1)
        xf = np.linspace(bins['mean_val'].min(), bins['mean_val'].max(), 50)
        ax.plot(xf, np.polyval(z, xf), '-', color='#e74c3c', linewidth=1.5, alpha=0.7, zorder=2)
        ax.text(0.05, 0.08, f'r={r:+.3f}\np={p:.3f}', transform=ax.transAxes, fontsize=7,
                va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
        if row == 0:
            ax.set_title(METRIC_LABELS[m], fontweight='bold')
        if row == 3:
            ax.set_xlabel('Distance')
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=8)
    axes[row, 0].text(-0.32, 1.08, panel_labels[row], transform=axes[row, 0].transAxes,
                      fontsize=16, fontweight='bold', va='top')

plt.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────
OUTPUT_DIR_PLOTS.mkdir(parents=True, exist_ok=True)
for suffix in ['.pdf', '.png']:
    fig.savefig(str(OUTPUT_DIR_PLOTS / f'supp_fig_s2dd_vs_sw{suffix}'), dpi=DPI, bbox_inches='tight')
fig.savefig(str(RESULTS_DIR / 's2dd_degradation' / 'supp_s2dd_vs_sw.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved supp_fig_s2dd_vs_sw.pdf/png")

# Save binned results
for name, bins in results.items():
    tag = name.split(' ')[0].replace('-', '_').lower()
    pd.DataFrame(bins).to_csv(RESULTS_DIR / 's2dd_degradation' / f'binned_{tag}.csv', index=False)

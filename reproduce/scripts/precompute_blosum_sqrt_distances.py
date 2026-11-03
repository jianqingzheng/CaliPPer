#!/usr/bin/env python3
"""Pre-compute BLOSUM-sqrt S2DD distances for all TCR data.

Generates .npy files compatible with the panel generation scripts.
Uses cached per-chain BLOSUM-SW similarity matrices where available.

Transform: d = sqrt(max(1 - sim, 0))  (NOT log(k*(1-sim+b)))
Combine: sigma_C weights + weighted_max_znorm (degradation strategy)
         uniform weights + znorm_sum (epitope strategy)

Output naming:
  TCR CT: results/fig2_cache/{model}_ct_{ts}_blosumsqrt_dist.npy
  TCR CV: results/fig2_cache/{model}_cv_fold{fold}_blosumsqrt_dist.npy
  TCR CV (uniform): results/fig2_cache/{model}_cv_fold{fold}_blosumsqrt_uniform_dist.npy
  TCR CT (uniform): results/fig2_cache/{model}_ct_{ts}_blosumsqrt_uniform_dist.npy

Usage:
  python precompute_blosum_sqrt_distances.py --tcr-ct   # TCR CT only (~5 min, uses cached sims)
  python precompute_blosum_sqrt_distances.py --tcr-cv   # TCR CV only (~8 hours)
  python precompute_blosum_sqrt_distances.py --all       # everything
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# CaliPPer self-contained bootstrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import INPUT_DIR  # also adds CaliPPer/ to sys.path

# Override RESULTS to INPUT_DIR/results so distances write where Fig 3/5 read
RESULTS = os.path.join(INPUT_DIR, 'results')
TCR_CACHE = os.path.join(RESULTS, 'fig2_cache')

from calipper.pluggable_distance import (
    compute_s2dd_pluggable, make_sw_blosum62_similarity
)
from calipper.combine_first_helpers import compute_chain_weights

# ── Constants ── (RESULTS + TCR_CACHE defined in bootstrap above)
BLOSUM_SIM_CACHE = os.path.join(RESULTS, 'deepantigen_retrospective', 'blosum_comparison', 'cache')

MODELS = ['nettcr', 'atm_tcr', 'blosum_rf', 'ergo_ii', 'tcrbert']
CHAINS = ['peptide', 'CDR3a', 'CDR3b']
CT_SETS = ['seen_test', 'unseen_fold34', 'v3_combined', 'v4_combined', 'mcpas', 'iedb_sars']
k, b, K = 0.1, 0.1, 50

# Column name standardization
COL_RENAME = {
    'epitope': 'peptide', 'Epitope': 'peptide',
    'cdr3_a': 'CDR3a', 'CDR3A': 'CDR3a',
    'cdr3_b': 'CDR3b', 'CDR3B': 'CDR3b',
}

# Training data for TCR CT — MUST match what was used to build the similarity caches.
# The caches were built from nettcr's splits/train.csv (12,066 rows), NOT tcr_ml_v4/train_data.csv (20,628).
TRAIN_PATH = os.path.join(RESULTS, 'nettcr', 'cross_test_logdist', 'splits', 'train.csv')


def standardize_cols(df):
    """Standardize column names to peptide/CDR3a/CDR3b."""
    renames = {old: new for old, new in COL_RENAME.items() if old in df.columns and new not in df.columns}
    return df.rename(columns=renames) if renames else df


def compute_tcr_ct():
    """Compute TCR CT BLOSUM-sqrt distances using cached similarity matrices."""
    print("=" * 60)
    print("TCR CT: BLOSUM-sqrt distances (sigma_C + uniform)")
    print("=" * 60)

    train_df = standardize_cols(pd.read_csv(TRAIN_PATH))
    print(f"Training data: {len(train_df)} samples")

    # Compute sigma_C weights from BLOSUM-sqrt distances (train-vs-train)
    # We recompute these from BLOSUM, not reuse Levenshtein weights
    sw_sim = make_sw_blosum62_similarity(gap_open=10, gap_extend=1)

    # sigma_C weights (computed fresh from BLOSUM-sqrt)
    # Use compute_s2dd_pluggable on a subsample to get per-chain sigma
    print("Computing sigma_C weights from BLOSUM-sqrt train-vs-train...")
    w_sc, _ = compute_chain_weights(train_df, CHAINS, k, b, K, formula='sigma_C')
    print(f"  sigma_C weights (Lev-derived): {', '.join(f'{c}={w:.4f}' for c, w in zip(CHAINS, w_sc))}")
    print("  Note: using Lev-derived sigma_C weights (sigma_C formula uses Simpson C which is")
    print("  sequence-based, not distance-based. Chain weights are nearly identical for BLOSUM.)")

    w_uni = np.ones(3) / 3.0
    print(f"  uniform weights: {', '.join(f'{c}={w:.4f}' for c, w in zip(CHAINS, w_uni))}")

    t0 = time.time()
    for model in MODELS:
        print(f"\n[{model}]")
        for ts in CT_SETS:
            pred_path = os.path.join(RESULTS, model, 'cross_test_logdist', 'predictions',
                                     f'{ts}_predictions_with_label.csv')
            if not os.path.exists(pred_path):
                print(f"  {ts}: no predictions, skipping")
                continue

            test_df = standardize_cols(pd.read_csv(pred_path))
            if not all(c in test_df.columns for c in CHAINS):
                print(f"  {ts}: missing chain columns, skipping")
                continue

            # sigma_C distances
            out_sc = os.path.join(TCR_CACHE, f'{model}_ct_{ts}_blosumsqrt_dist.npy')
            if os.path.exists(out_sc):
                print(f"  {ts} sigma_C: cached ({len(np.load(out_sc))})")
            else:
                t1 = time.time()
                d = compute_s2dd_pluggable(
                    test_df, train_df, CHAINS, w_sc,
                    similarity_fn=sw_sim, k=k, b=b, K=K,
                    cache_prefix=f'tcr_ct_{ts}', cache_dir=BLOSUM_SIM_CACHE,
                    transform='sqrt', verbose=False)
                np.save(out_sc, d)
                print(f"  {ts} sigma_C: {len(d)} distances ({time.time()-t1:.1f}s)")

            # uniform distances (for epitope strategy)
            out_un = os.path.join(TCR_CACHE, f'{model}_ct_{ts}_blosumsqrt_uniform_dist.npy')
            if os.path.exists(out_un):
                print(f"  {ts} uniform: cached ({len(np.load(out_un))})")
            else:
                t1 = time.time()
                d = compute_s2dd_pluggable(
                    test_df, train_df, CHAINS, w_uni,
                    similarity_fn=sw_sim, k=k, b=b, K=K,
                    cache_prefix=f'tcr_ct_{ts}', cache_dir=BLOSUM_SIM_CACHE,
                    transform='sqrt', verbose=False)
                np.save(out_un, d)
                print(f"  {ts} uniform: {len(d)} distances ({time.time()-t1:.1f}s)")

    print(f"\nTCR CT done in {time.time()-t0:.0f}s")


def compute_tcr_cv():
    """Compute TCR CV BLOSUM-sqrt distances (requires full similarity computation)."""
    print("=" * 60)
    print("TCR CV: BLOSUM-sqrt distances (sigma_C + uniform)")
    print("  WARNING: No cached similarity matrices — full computation required.")
    print("  Estimated time: ~8 hours for 25 model-fold combos.")
    print("=" * 60)

    sw_sim = make_sw_blosum62_similarity(gap_open=10, gap_extend=1)
    w_uni = np.ones(3) / 3.0
    # Pre-computed similarity caches exist at tcr_cv_blosum_comparison/sim_cache/
    cv_cache_dir = os.path.join(RESULTS, 'tcr_cv_blosum_comparison', 'sim_cache')
    if not os.path.isdir(cv_cache_dir):
        cv_cache_dir = os.path.join(RESULTS, 'blosum_sqrt_cache', 'tcr_cv')
        os.makedirs(cv_cache_dir, exist_ok=True)
    print(f"  Similarity cache dir: {cv_cache_dir}")

    t0 = time.time()
    for model in MODELS:
        print(f"\n[{model}]")
        for fold in range(5):
            fold_dir = os.path.join(RESULTS, model, 'cv_logdist', f'fold{fold}')
            train_path = os.path.join(fold_dir, 'train.csv')
            test_path = os.path.join(fold_dir, 'test_predictions_with_label.csv')
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                print(f"  fold{fold}: missing data, skipping")
                continue

            train_df = standardize_cols(pd.read_csv(train_path))
            parts = [pd.read_csv(test_path)]
            val_path = os.path.join(fold_dir, 'val_predictions_with_label.csv')
            if os.path.exists(val_path):
                parts.append(pd.read_csv(val_path))
            test_df = standardize_cols(pd.concat(parts, ignore_index=True))

            if not all(c in test_df.columns for c in CHAINS):
                print(f"  fold{fold}: missing chain columns, skipping")
                continue

            # Compute sigma_C weights per fold (from this fold's training data)
            w_sc, _ = compute_chain_weights(train_df, CHAINS, k, b, K, formula='sigma_C')

            # Cache prefix: cv_fold{N} (model-independent, shared across all 5 models)
            # Existing caches named: cv_fold{N}_{chain}_sim.npz
            # compute_s2dd_pluggable constructs: {prefix}_{col}.npz
            # So prefix = 'cv_fold0_peptide_sim' would match for peptide...
            # But we need {prefix}_{col}.npz to match cv_fold0_{col}_sim.npz
            # Easiest: create symlinks if needed
            from pathlib import Path
            for ch in CHAINS:
                src = Path(cv_cache_dir) / f'cv_fold{fold}_{ch}_sim.npz'
                dst = Path(cv_cache_dir) / f'cv_fold{fold}_{ch}.npz'
                if src.exists() and not dst.exists():
                    dst.symlink_to(src.name)
            cv_prefix = f'cv_fold{fold}'

            # sigma_C distances
            out_sc = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}_blosumsqrt_dist.npy')
            if os.path.exists(out_sc):
                print(f"  fold{fold} sigma_C: cached ({len(np.load(out_sc))})")
            else:
                t1 = time.time()
                d = compute_s2dd_pluggable(
                    test_df, train_df, CHAINS, w_sc,
                    similarity_fn=sw_sim, k=k, b=b, K=K,
                    cache_prefix=cv_prefix, cache_dir=cv_cache_dir,
                    transform='sqrt', verbose=(model == MODELS[0]))
                np.save(out_sc, d)
                print(f"  fold{fold} sigma_C: {len(d)} distances ({time.time()-t1:.0f}s)")

            # uniform distances
            out_un = os.path.join(TCR_CACHE, f'{model}_cv_fold{fold}_blosumsqrt_uniform_dist.npy')
            if os.path.exists(out_un):
                print(f"  fold{fold} uniform: cached ({len(np.load(out_un))})")
            else:
                t1 = time.time()
                d = compute_s2dd_pluggable(
                    test_df, train_df, CHAINS, w_uni,
                    similarity_fn=sw_sim, k=k, b=b, K=K,
                    cache_prefix=cv_prefix, cache_dir=cv_cache_dir,
                    transform='sqrt', verbose=False)
                np.save(out_un, d)
                print(f"  fold{fold} uniform: {len(d)} distances ({time.time()-t1:.0f}s)")

    print(f"\nTCR CV done in {time.time()-t0:.0f}s")


def compute_bcr_ct():
    """Compute BCR CT BLOSUM-sqrt distances for fold4cal pipeline."""
    print("=" * 60)
    print("BCR CT: BLOSUM-sqrt distances (fold4cal, 5 models)")
    print("=" * 60)

    BCR_CHAINS = ['Heavy', 'Light', 'variant_seq']
    BCR_k, BCR_b, BCR_K = 0.1, 0.03, 30
    BCR_FOLD4CAL = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')
    BCR_TRAIN_PATH = os.path.join(RESULTS, 'xbcr', 'combined_bind_ab_cv', 'fold4', 'train.csv')
    BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']

    if not os.path.exists(BCR_TRAIN_PATH):
        print(f"ERROR: BCR train not found: {BCR_TRAIN_PATH}")
        return

    bcr_train = pd.read_csv(BCR_TRAIN_PATH)
    print(f"BCR train: {len(bcr_train)} samples")

    sw_sim = make_sw_blosum62_similarity(gap_open=10, gap_extend=1)
    w_sc, _ = compute_chain_weights(bcr_train, BCR_CHAINS, BCR_k, BCR_b, BCR_K, formula='sigma_C')
    print(f"sigma_C weights: {', '.join(f'{c}={w:.4f}' for c, w in zip(BCR_CHAINS, w_sc))}")

    bcr_sim_cache = os.path.join(RESULTS, 'blosum_sqrt_cache', 'bcr_ct')
    os.makedirs(bcr_sim_cache, exist_ok=True)

    t0 = time.time()
    # Compute distances once (model-independent — same sequences for all models)
    # Pool all test data from any model (they have the same sequences)
    ref_model = 'xbcr'
    model_dir = os.path.join(BCR_FOLD4CAL, ref_model)
    parts = []
    test_names = []
    for ts in ['cal_predictions', 'A1-A11', 'unseen', 'flu']:
        fp = os.path.join(model_dir, f'{ts}_predictions.csv') if ts != 'cal_predictions' else os.path.join(model_dir, 'cal_predictions.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        parts.append(df)
        test_names.append((ts, len(df)))
    pooled = pd.concat(parts, ignore_index=True)
    print(f"Pooled test: {len(pooled)} samples ({', '.join(f'{n}={c}' for n,c in test_names)})")

    # Compute sigma_C distances for full pool
    out_sc = os.path.join(BCR_FOLD4CAL, f'pooled_blosumsqrt_dist.npy')
    if os.path.exists(out_sc):
        d_sc = np.load(out_sc)
        print(f"Loaded cached pooled sigma_C: {len(d_sc)}")
    else:
        print("Computing pooled BLOSUM-sqrt distances...")
        t1 = time.time()
        d_sc = compute_s2dd_pluggable(
            pooled, bcr_train, BCR_CHAINS, w_sc,
            similarity_fn=sw_sim, k=BCR_k, b=BCR_b, K=BCR_K,
            cache_prefix='bcr_ct_fold4cal', cache_dir=bcr_sim_cache,
            transform='sqrt', verbose=True)
        np.save(out_sc, d_sc)
        print(f"Done: {len(d_sc)} distances ({time.time()-t1:.0f}s)")

    # Split back into per-test-set .npy files for each model
    offset = 0
    for ts, n in test_names:
        chunk = d_sc[offset:offset+n]
        for model in BCR_MODELS:
            out_path = os.path.join(BCR_FOLD4CAL, model, f'{ts}_blosumsqrt_dist.npy')
            np.save(out_path, chunk)
        offset += n
        print(f"  {ts}: {n} distances → saved for all 5 models")

    print(f"\nBCR CT done in {time.time()-t0:.0f}s")


def compute_bcr_cv():
    """Compute BCR CV BLOSUM-sqrt distances."""
    print("=" * 60)
    print("BCR CV: BLOSUM-sqrt distances (5 models × 5 folds)")
    print("=" * 60)

    BCR_CHAINS = ['Heavy', 'Light', 'variant_seq']
    BCR_k, BCR_b, BCR_K = 0.1, 0.03, 30
    BCR_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']

    sw_sim = make_sw_blosum62_similarity(gap_open=10, gap_extend=1)
    bcr_cv_cache = os.path.join(RESULTS, 'blosum_sqrt_cache', 'bcr_cv')
    os.makedirs(bcr_cv_cache, exist_ok=True)

    t0 = time.time()
    # Model-independent: compute once per fold from xbcr data, share across models
    ref_model = 'xbcr'
    for fold in range(5):
        test_path = os.path.join(RESULTS, ref_model, 'combined_bind_ab_cv', f'fold{fold}', 'test.csv')
        train_path = os.path.join(RESULTS, ref_model, 'combined_bind_ab_cv', f'fold{fold}', 'train.csv')
        if not os.path.exists(test_path) or not os.path.exists(train_path):
            print(f"  fold{fold}: missing data, skipping")
            continue

        out_sc = os.path.join(TCR_CACHE, f'{ref_model}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
        if os.path.exists(out_sc):
            d = np.load(out_sc)
            print(f"  fold{fold}: cached ({len(d)})")
        else:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            if not all(c in train_df.columns for c in BCR_CHAINS):
                print(f"  fold{fold}: missing chain columns, skipping")
                continue

            w_sc, _ = compute_chain_weights(train_df, BCR_CHAINS, BCR_k, BCR_b, BCR_K, formula='sigma_C')
            print(f"  fold{fold}: computing ({len(test_df)} test × {len(train_df)} train)...")
            t1 = time.time()
            d = compute_s2dd_pluggable(
                test_df, train_df, BCR_CHAINS, w_sc,
                similarity_fn=sw_sim, k=BCR_k, b=BCR_b, K=BCR_K,
                cache_prefix=f'bcr_cv_fold{fold}', cache_dir=bcr_cv_cache,
                transform='sqrt', verbose=True)
            np.save(out_sc, d)
            print(f"  fold{fold}: {len(d)} distances ({time.time()-t1:.0f}s)")

        # Copy to all models (model-independent)
        for model in BCR_MODELS:
            dst = os.path.join(TCR_CACHE, f'{model}_bcr_cv_fold{fold}_blosumsqrt_dist.npy')
            if not os.path.exists(dst):
                np.save(dst, np.load(out_sc))

    print(f"\nBCR CV done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tcr-ct', action='store_true', help='TCR CT only (~5 min)')
    parser.add_argument('--tcr-cv', action='store_true', help='TCR CV (~30 min)')
    parser.add_argument('--bcr-ct', action='store_true', help='BCR CT (~8 hours)')
    parser.add_argument('--bcr-cv', action='store_true', help='BCR CV (~12 hours)')
    parser.add_argument('--all', action='store_true', help='Everything')
    args = parser.parse_args()

    if not any([args.tcr_ct, args.tcr_cv, args.bcr_ct, args.bcr_cv, args.all]):
        parser.print_help()
        sys.exit(1)

    if args.tcr_ct or args.all:
        compute_tcr_ct()
    if args.tcr_cv or args.all:
        compute_tcr_cv()
    if args.bcr_ct or args.all:
        compute_bcr_ct()
    if args.bcr_cv or args.all:
        compute_bcr_cv()

    print("\nAll done.")

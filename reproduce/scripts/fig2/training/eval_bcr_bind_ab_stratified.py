#!/usr/bin/env python3
"""
BCR Binding — Antibody-Stratified CV with XBCR-net (SARS-only, 1-pathogen).

⚠ STAGED AS A HELPER LIBRARY ONLY (NOT a retraining target).

Per memory feedback_bcr_cv_3pathogen_vs_2pathogen.md, the canonical BCR
training uses 2-pathogen (SARS+flu) combined binding, NOT SARS-only.
This script is the original 1-pathogen variant and is NOT a Fig 2-5
reproduction target — it is staged here ONLY because the canonical
2-pathogen scripts (eval_bcr_bind_ct_fold4cal.py + eval_bcr_combined_ab_
stratified.py) import HELPER FUNCTIONS from this module (prepare_xbcrnet_data,
train_xbcrnet_fold, infer_xbcrnet_fold, collect_predictions).

DO NOT invoke this script directly as a training pipeline — it is
intentionally NOT in retrain_fig3_inputs.sh's MODELS registry. Use:
  - retrain_fig3_inputs.sh --model bcr_ct_fold4cal   (canonical CT)
  - retrain_fig3_inputs.sh --model bcr_cv_combined   (canonical CV)

Original (1-pathogen) docstring follows for historical reference:

Full pipeline: pool binding data from 5 XBCR-net prediction sources → create
antibody-stratified 5-fold CV splits (unique Heavy+Light per fold, all/most
variants seen) → prepare XBCR-net training data per fold → train XBCR-net →
run inference → collect predictions → evaluate LogDist degradation.

Differs from eval_bcr_cv_freqw_comparison.py (variant-stratified) in that:
- Splits are by unique (Heavy, Light) antibody pairs, not by variant
- All/most variants appear in both train and test sets
- Scoping uses seen-Heavy / unseen-Heavy (not variant_seen)
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from General_Eval.general_evaluator import (
    safe_metric,
    binned_correlations,
)
from General_Eval.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances,
)

# Use tf env for GPU-enabled TensorFlow (XBCR-net training/inference)
TF_PYTHON = '/home/jzheng/anaconda3/envs/tf/bin/python'


def _flush():
    sys.stdout.flush()


# ============================================================================
# Step 1: Load binding data
# ============================================================================

def load_binding_data(data_root):
    """Load and pool all BCR binding prediction files.

    Pools 5 sources from Data/bcr_seq/XBCR_net_binding/.
    Drops pred_prob since it will be regenerated per fold after retraining.

    Includes not_rbd column where available. For sources without not_rbd,
    it is inferred as (1 - rbd). Samples with both rbd=0 and not_rbd=0
    are filtered out (ambiguous).

    Returns a DataFrame with columns:
        Heavy, Light, variant_seq, rbd, not_rbd, source, ab
    """
    bcr_dir = os.path.join(data_root, 'Data', 'bcr_seq', 'XBCR_net_binding')

    sources = {
        'xbcr_train_folder': 'xbcr_train',
        'A1-A11_testset': 'A1-A11',
        'unseen_testset': 'unseen',
        'guoyu_testset': 'guoyu',
        'BNT162b2_testset': 'BNT162b2',
    }

    frames = []
    for folder, label in sources.items():
        path = os.path.join(bcr_dir, folder, 'results_rbd_XBCR_net-0.xlsx')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue

        df = pd.read_excel(path)

        # Extract not_rbd if available, otherwise infer from rbd
        if 'not_rbd' in df.columns:
            not_rbd = df['not_rbd'].astype(int)
        else:
            not_rbd = (1 - df['rbd']).astype(int)

        sub = pd.DataFrame({
            'Heavy': df['Heavy'].astype(str),
            'Light': df['Light'].astype(str),
            'variant_seq': df['variant_seq'].astype(str),
            'rbd': df['rbd'].astype(int),
            'not_rbd': not_rbd,
            'source': label,
        })

        # Filter out ambiguous samples (rbd=0 AND not_rbd=0)
        ambiguous = (sub['rbd'] == 0) & (sub['not_rbd'] == 0)
        n_ambig = ambiguous.sum()
        if n_ambig > 0:
            print(f"  WARNING: {label}: filtering {n_ambig} ambiguous samples "
                  f"(rbd=0, not_rbd=0)")
            sub = sub[~ambiguous].reset_index(drop=True)

        frames.append(sub)
        n_rbd = (sub['rbd'] == 1).sum()
        n_notrbd = (sub['not_rbd'] == 1).sum()
        print(f"  Loaded {label}: {len(sub)} rows, "
              f"{sub['variant_seq'].nunique()} variants, "
              f"rbd={n_rbd}, not_rbd={n_notrbd}")

    pooled = pd.concat(frames, ignore_index=True)

    # Clean sequences
    for col in ['Heavy', 'Light', 'variant_seq']:
        pooled[col] = pooled[col].str.replace(r'[\s_|><=-]', '', regex=True)

    # Antibody identity
    pooled['ab'] = pooled['Heavy'] + '|' + pooled['Light']
    n_abs = pooled['ab'].nunique()

    n_rbd = (pooled['rbd'] == 1).sum()
    n_notrbd = (pooled['not_rbd'] == 1).sum()
    print(f"\n  Total pooled: {len(pooled)} samples, "
          f"{pooled['variant_seq'].nunique()} unique variants, "
          f"{n_abs} unique antibodies (Heavy+Light)")
    print(f"  RBD binders: {n_rbd}, non-RBD binders: {n_notrbd}")
    return pooled


# ============================================================================
# Step 2: Antibody-stratified CV splits
# ============================================================================

def create_antibody_stratified_splits(pooled_df, n_folds=5, seed=42):
    """Create antibody-stratified CV folds.

    - Splits by unique (Heavy, Light) pairs so no antibody is shared
      between train and test
    - All/most variants appear in both train and test (warned if missing)

    Returns:
        ab_assignments: DataFrame (ab, fold, n_samples)
        folds: list of (train_df, test_df) tuples
    """
    rng = np.random.RandomState(seed)

    # Get unique antibodies sorted by sample count (descending for balance)
    ab_counts = pooled_df.groupby('ab').size().reset_index(name='n_samples')
    ab_counts = ab_counts.sort_values('n_samples', ascending=False).reset_index(drop=True)

    # Shuffle then round-robin assign for balanced fold sizes
    unique_abs = ab_counts['ab'].tolist()
    rng.shuffle(unique_abs)

    fold_assignments = {}
    for i, ab in enumerate(unique_abs):
        fold_assignments[ab] = i % n_folds

    ab_counts['fold'] = ab_counts['ab'].map(fold_assignments)

    # Build train/test splits
    pooled_df = pooled_df.copy()
    pooled_df['fold'] = pooled_df['ab'].map(fold_assignments)

    folds = []

    for k in range(n_folds):
        test_mask = pooled_df['fold'] == k
        train_df = pooled_df[~test_mask].reset_index(drop=True)
        test_df = pooled_df[test_mask].reset_index(drop=True)

        # Check variant coverage (warn, don't assert)
        train_variants = set(train_df['variant_seq'].unique())
        test_variants = set(test_df['variant_seq'].unique())
        missing_in_train = test_variants - train_variants
        if missing_in_train:
            print(f"  WARNING fold {k}: {len(missing_in_train)} test variants "
                  f"NOT in training (rare variants with few samples)")

        # variant_seen: informational
        test_df['variant_seen'] = test_df['variant_seq'].apply(
            lambda v: 1 if v in train_variants else 0)

        # Verify no antibody overlap
        train_abs = set(train_df['ab'].unique())
        test_abs = set(test_df['ab'].unique())
        overlap = train_abs & test_abs
        assert len(overlap) == 0, \
            f"Fold {k}: {len(overlap)} antibodies in both train and test!"

        # Heavy chain seen/unseen flag
        train_heavies = set(train_df['Heavy'].unique())
        test_df['heavy_seen'] = test_df['Heavy'].apply(
            lambda h: 1 if h in train_heavies else 0)

        folds.append((train_df, test_df))

    print(f"\n  Antibody-stratified {n_folds}-fold splits:")
    for k, (tr, te) in enumerate(folds):
        n_var_seen = (te['variant_seen'] == 1).sum()
        n_var_unseen = (te['variant_seen'] == 0).sum()
        n_heavy_seen = (te['heavy_seen'] == 1).sum()
        n_heavy_unseen = (te['heavy_seen'] == 0).sum()
        tr_vars = tr['variant_seq'].nunique()
        te_vars = te['variant_seq'].nunique()
        tr_abs = tr['ab'].nunique()
        te_abs = te['ab'].nunique()
        print(f"    Fold {k}: train={len(tr)} ({tr_abs} abs, {tr_vars} var), "
              f"test={len(te)} ({te_abs} abs, {te_vars} var), "
              f"heavy_seen={n_heavy_seen}, heavy_unseen={n_heavy_unseen}, "
              f"pos_rate={te['rbd'].mean():.3f}")

    return ab_counts[['ab', 'fold', 'n_samples']], folds


# ============================================================================
# Step 3: Prepare XBCR-net training data per fold
# ============================================================================

def prepare_xbcrnet_data(fold_idx, train_df, test_df, xbcrnet_dir,
                         data_prefix='bind_ab'):
    """Create the directory structure XBCR-net expects for one fold.

    exper/: All training samples with rbd=1 (positive) or not_rbd=1 (negative).
            The model uses label = rbd AND NOT not_rbd internally.
    nonexp/: Disease-unrelated antibodies (HIV, RSV, etc.) from the original
             XBCR-net binding dataset — assumed negatives.
    """
    import shutil

    data_name = f'{data_prefix}_fold{fold_idx}'
    data_dir = os.path.join(xbcrnet_dir, 'data', data_name)

    for subdir in ['exper', 'nonexp',
                   os.path.join('test', 'ab_to_pred'),
                   os.path.join('test', 'ag_to_pred'),
                   os.path.join('test', 'results')]:
        os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)

    # Experimental training data: both RBD binders (rbd=1) and non-RBD binders
    # (not_rbd=1). XBCR-net computes label = all([rbd, 1 - not_rbd]).
    exper_df = train_df[(train_df['rbd'] == 1) |
                        (train_df['not_rbd'] == 1)].copy()
    exper_out = exper_df[['Heavy', 'Light', 'variant_seq', 'rbd', 'not_rbd']]
    exper_path = os.path.join(data_dir, 'exper', 'train_exper.xlsx')
    exper_out.to_excel(exper_path, index=False)
    n_rbd = (exper_df['rbd'] == 1).sum()
    n_notrbd = (exper_df['not_rbd'] == 1).sum()
    print(f"    Exper training: {len(exper_out)} samples "
          f"(rbd={n_rbd}, not_rbd={n_notrbd}) -> {exper_path}")

    # Non-experimental negatives: copy original disease-unrelated antibodies
    orig_nonexp_dir = os.path.join(xbcrnet_dir, 'data', 'binding', 'nonexp')
    fold_nonexp_dir = os.path.join(data_dir, 'nonexp')
    n_copied = 0
    for fname in os.listdir(orig_nonexp_dir):
        if fname.endswith('.xlsx'):
            src = os.path.join(orig_nonexp_dir, fname)
            dst = os.path.join(fold_nonexp_dir, fname)
            shutil.copy2(src, dst)
            df_neg = pd.read_excel(src)
            n_copied += len(df_neg)
            print(f"    Nonexp: copied {fname} ({len(df_neg)} rows) -> {dst}")

    # Test antibodies
    test_ab = test_df[['Heavy', 'Light']].drop_duplicates().reset_index(drop=True)
    test_ab['Name'] = [f'ab_{i}' for i in range(len(test_ab))]
    ab_path = os.path.join(data_dir, 'test', 'ab_to_pred', 'test_antibodies.xlsx')
    test_ab.to_excel(ab_path, index=False)
    print(f"    Test antibodies: {len(test_ab)} unique -> {ab_path}")

    # Test antigens
    MAX_SEQ_LEN = 600  # increased from 290 to support flu HA (~566 AA)
    test_ag = test_df[['variant_seq']].drop_duplicates().reset_index(drop=True)
    long_mask = test_ag['variant_seq'].str.len() > MAX_SEQ_LEN
    if long_mask.any():
        n_long = long_mask.sum()
        print(f"    WARNING: {n_long} antigen(s) exceed {MAX_SEQ_LEN} AA, truncating")
        test_ag.loc[long_mask, 'variant_seq'] = \
            test_ag.loc[long_mask, 'variant_seq'].str[:MAX_SEQ_LEN]
    test_ag['variant_name'] = [f'var_{i}' for i in range(len(test_ag))]
    test_ag['rbd'] = 1
    ag_path = os.path.join(data_dir, 'test', 'ag_to_pred', 'test_antigens.xlsx')
    test_ag.to_excel(ag_path, index=False)
    print(f"    Test antigens: {len(test_ag)} unique -> {ag_path}")

    # Paired test file: actual (antibody, antigen) pairs for direct inference
    # Avoids the N_ab × N_ag cross-product (e.g. 1.4M → 12K predictions)
    paired_dir = os.path.join(data_dir, 'test', 'paired')
    os.makedirs(paired_dir, exist_ok=True)
    paired_df = test_df[['Heavy', 'Light', 'variant_seq', 'rbd']].copy()
    if long_mask is not None:
        # Truncate variant_seq in paired data too
        long_paired = paired_df['variant_seq'].str.len() > MAX_SEQ_LEN
        if long_paired.any():
            paired_df.loc[long_paired, 'variant_seq'] = \
                paired_df.loc[long_paired, 'variant_seq'].str[:MAX_SEQ_LEN]
    paired_df['variant_name'] = [f'pair_{i}' for i in range(len(paired_df))]
    # Set rbd=1 for all samples so data_process doesn't filter negatives
    paired_df['rbd'] = 1
    paired_df['not_rbd'] = 0
    paired_path = os.path.join(paired_dir, 'test_paired.xlsx')
    paired_df.to_excel(paired_path, index=False)
    print(f"    Paired test: {len(paired_df)} samples -> {paired_path}")

    return data_name


# ============================================================================
# Step 4 & 5: Train and infer XBCR-net per fold
# ============================================================================

def _copy_pretrained_model(xbcrnet_dir, data_name, model_num,
                           source_data='binding', source_num=0):
    """Copy pretrained binding model weights to new data_name model directory."""
    import shutil
    src_dir = os.path.join(xbcrnet_dir, 'models', source_data,
                           f'{source_data}-XBCR_net')
    dst_dir = os.path.join(xbcrnet_dir, 'models', data_name,
                           f'{data_name}-XBCR_net')
    os.makedirs(dst_dir, exist_ok=True)

    src_prefix = f'model_rbd_{source_num}.tf'
    dst_prefix = f'model_rbd_{model_num}.tf'

    copied = []
    for fname in os.listdir(src_dir):
        if fname.startswith(src_prefix):
            suffix = fname[len(src_prefix):]
            dst_fname = dst_prefix + suffix
            shutil.copy2(os.path.join(src_dir, fname),
                         os.path.join(dst_dir, dst_fname))
            copied.append(dst_fname)

    ckpt_path = os.path.join(dst_dir, 'checkpoint')
    with open(ckpt_path, 'w') as f:
        f.write(f'model_checkpoint_path: "{dst_prefix}"\n')
        f.write(f'all_model_checkpoint_paths: "{dst_prefix}"\n')
    copied.append('checkpoint')

    print(f"    Copied pretrained model: {src_dir} -> {dst_dir}")
    print(f"    Files: {copied}")
    return dst_dir


def train_xbcrnet_fold(fold_idx, xbcrnet_dir, max_epochs, restore_pretrain=1,
                       data_prefix='bind_ab'):
    """Train XBCR-net on one fold's data."""
    data_name = f'{data_prefix}_fold{fold_idx}'

    if restore_pretrain:
        model_dir = os.path.join(xbcrnet_dir, 'models', data_name,
                                 f'{data_name}-XBCR_net')
        expected = os.path.join(model_dir, f'model_rbd_{fold_idx}.tf.index')
        if not os.path.exists(expected):
            _copy_pretrained_model(xbcrnet_dir, data_name, fold_idx)

    cmd = [
        TF_PYTHON, 'main_train.py',
        '--data_name', data_name,
        '--model_num', str(fold_idx),
        '--max_epochs', str(max_epochs),
        '--include_light', '1',
        '--restore_pretrain', str(restore_pretrain),
    ]
    print(f"  Training fold {fold_idx}: {' '.join(cmd)}")
    _flush()

    result = subprocess.run(
        cmd, cwd=xbcrnet_dir,
        capture_output=True, text=True, timeout=28800)

    if result.returncode != 0:
        print(f"  ERROR training fold {fold_idx}:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        _flush()
        return False

    lines = result.stdout.strip().split('\n')
    for line in lines[-5:]:
        print(f"    {line}")
    _flush()
    return True


def infer_xbcrnet_fold(fold_idx, xbcrnet_dir, data_prefix='bind_ab'):
    """Run XBCR-net inference on one fold's test data.

    Uses paired mode (test/paired/) if available — predicts only actual
    (antibody, antigen) pairs instead of the full cross-product.
    Falls back to cross-product mode (ab_to_pred × ag_to_pred) if not.
    """
    data_name = f'{data_prefix}_fold{fold_idx}'

    model_dir = os.path.join(xbcrnet_dir, 'models', data_name,
                             f'{data_name}-XBCR_net')
    expected = os.path.join(model_dir, f'model_rbd_{fold_idx}.tf.index')
    if not os.path.exists(expected):
        print(f"  ERROR: model not found at {expected}")
        return None

    # Use paired mode if paired test file exists (avoids N×M cross-product)
    paired_dir = os.path.join(xbcrnet_dir, 'data', data_name, 'test', 'paired')
    use_paired = os.path.exists(os.path.join(paired_dir, 'test_paired.xlsx'))

    cmd = [
        TF_PYTHON, 'main_infer.py',
        '--data_name', data_name,
        '--model_num', str(fold_idx),
        '--include_light', '1',
    ]
    if use_paired:
        cmd.extend(['--paired', '1'])
    print(f"  Inference fold {fold_idx} ({'paired' if use_paired else 'cross-product'}): "
          f"{' '.join(cmd)}")
    _flush()

    result = subprocess.run(
        cmd, cwd=xbcrnet_dir,
        capture_output=True, text=True, timeout=86400)

    if result.returncode != 0:
        print(f"  ERROR inference fold {fold_idx}:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        _flush()
        return None

    result_base = os.path.join(
        xbcrnet_dir, 'data', data_name, 'test', 'results',
        f'results_rbd_XBCR_net-{fold_idx}')
    # Try CSV first (needed for >1M rows), then xlsx
    if os.path.exists(result_base + '.csv'):
        result_path = result_base + '.csv'
        pred_df = pd.read_csv(result_path)
    elif os.path.exists(result_base + '.xlsx'):
        result_path = result_base + '.xlsx'
        pred_df = pd.read_excel(result_path)
    else:
        print(f"  ERROR: expected output not found at {result_base}.[csv|xlsx]")
        _flush()
        return None
    print(f"    Predictions: {len(pred_df)} rows from {result_path}")
    _flush()
    return pred_df


# ============================================================================
# Step 6: Collect predictions
# ============================================================================

def collect_predictions(fold_idx, pred_df, test_df, train_df):
    """Match XBCR-net predictions back to original test samples.

    In paired mode, pred_df has the same rows as test_df (1:1 match).
    In cross-product mode, matches by (Heavy, Light, variant_seq[:290]).
    Also computes the heavy_seen flag per test sample.
    """
    for col in ['Heavy', 'Light', 'variant_seq']:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col].astype(str).str.replace(
                r'[\s_|><=-]', '', regex=True)

    # Check if paired mode (pred_df size ≈ test_df size)
    if abs(len(pred_df) - len(test_df)) < 100:
        # Paired mode: direct 1:1 assignment
        test_out = test_df.copy()
        test_out['pred_prob'] = pred_df['pred_prob'].values[:len(test_out)]
        n_matched = (~test_out['pred_prob'].isna()).sum()
        n_missing = test_out['pred_prob'].isna().sum()
        print(f"    Fold {fold_idx}: paired mode, {n_matched} predictions assigned, "
              f"{n_missing} missing")
        # heavy_seen flag
        train_heavies = set(train_df['Heavy'].unique())
        test_out['heavy_seen'] = test_out['Heavy'].apply(
            lambda h: 1 if h in train_heavies else 0)
        test_out['variant_seen'] = test_out['variant_seq'].apply(
            lambda v: 1 if v in set(train_df['variant_seq'].unique()) else 0)
        n_heavy_seen = (test_out['heavy_seen'] == 1).sum()
        n_heavy_unseen = (test_out['heavy_seen'] == 0).sum()
        print(f"    Heavy chain: {n_heavy_seen} seen, {n_heavy_unseen} unseen")
        return test_out

    MAX_SEQ_LEN = 600  # increased from 290 to support flu HA (~566 AA)
    pred_lookup = {}
    for _, row in pred_df.iterrows():
        key = (str(row['Heavy']), str(row['Light']),
               str(row['variant_seq'])[:MAX_SEQ_LEN])
        pred_lookup[key] = float(row['pred_prob'])

    test_out = test_df.copy()
    preds = []
    n_matched = 0
    for _, row in test_out.iterrows():
        key = (str(row['Heavy']), str(row['Light']),
               str(row['variant_seq'])[:MAX_SEQ_LEN])
        if key in pred_lookup:
            preds.append(pred_lookup[key])
            n_matched += 1
        else:
            preds.append(np.nan)

    test_out['pred_prob'] = preds
    n_missing = test_out['pred_prob'].isna().sum()

    print(f"    Fold {fold_idx}: matched {n_matched}/{len(test_out)}, "
          f"missing {n_missing}")

    if n_missing > 0:
        print(f"    WARNING: {n_missing} test samples not found in predictions")
        test_out = test_out.dropna(subset=['pred_prob']).reset_index(drop=True)

    # Recompute heavy_seen flag (in case test_out was filtered)
    train_heavies = set(train_df['Heavy'].unique())
    test_out['heavy_seen'] = test_out['Heavy'].apply(
        lambda h: 1 if h in train_heavies else 0)

    n_heavy_seen = (test_out['heavy_seen'] == 1).sum()
    n_heavy_unseen = (test_out['heavy_seen'] == 0).sum()
    print(f"    Heavy chain: {n_heavy_seen} seen, {n_heavy_unseen} unseen")

    return test_out


# ============================================================================
# Step 7: LogDist degradation analysis
# ============================================================================

## sigma_H replaced by sigma_C from combine_first_helpers.py


def evaluate_fold_degradation(fold_idx, train_df, test_df, chain_cols,
                              k, b, K, bin_num, eval_metrics, seed=42):
    """Compute LogDist and evaluate degradation for one fold.

    Scopes: seen_heavy, unseen_heavy, combined.
    """
    n_test = len(test_df)
    n_heavy_seen = (test_df['heavy_seen'] == 1).sum() \
        if 'heavy_seen' in test_df.columns else 0
    n_heavy_unseen = (test_df['heavy_seen'] == 0).sum() \
        if 'heavy_seen' in test_df.columns else 0
    print(f"\n  Fold {fold_idx}: train={len(train_df)}, test={n_test} "
          f"(heavy_seen={n_heavy_seen}, heavy_unseen={n_heavy_unseen})")

    print(f"  Computing sigma_C weights...")
    weights, _ = compute_chain_weights(train_df, chain_cols, k, b, K,
                                        formula='sigma_C', subsample=500, seed=seed)
    w_str = ', '.join(f'{c}={weights[i]:.4f}'
                      for i, c in enumerate(chain_cols))
    print(f"  Weights: {w_str}")

    print(f"  Computing combine-first weighted_max_znorm distances...")
    topk_combined = compute_combine_first_distances(
        test_df, train_df, chain_cols, weights, k, b, K)

    rows = []
    method = 'topk'
    base_df = pd.DataFrame({
        'label': test_df['rbd'].astype(float),
        'pred': test_df['pred_prob'].astype(float),
        'distance': topk_combined,
    })
    if 'heavy_seen' in test_df.columns:
        base_df['heavy_seen'] = test_df['heavy_seen'].astype(int)

    seen_heavy_df = base_df[base_df['heavy_seen'] == 1] \
        if 'heavy_seen' in base_df.columns else pd.DataFrame()
    unseen_heavy_df = base_df[base_df['heavy_seen'] == 0] \
        if 'heavy_seen' in base_df.columns else pd.DataFrame()

    scopes = [('combined', base_df)]
    if len(seen_heavy_df) > 0:
        scopes.append(('seen_heavy', seen_heavy_df))
    if len(unseen_heavy_df) > 0:
        scopes.append(('unseen_heavy', unseen_heavy_df))

    for scope_name, scope_df in scopes:
        dist_unique = scope_df['distance'].nunique()
        if len(scope_df) >= bin_num * 2 and dist_unique >= bin_num:
            binned = binned_correlations(
                scope_df, 'distance', eval_metrics, bin_num)
        else:
            binned = {
                m: {'pearson_r': np.nan, 'pearson_p': np.nan,
                     'spearman_r': np.nan, 'spearman_p': np.nan,
                     'bin_dists': [], 'bin_perfs': []}
                for m in eval_metrics}

        for m in eval_metrics:
            rows.append({
                'fold': fold_idx,
                'method': method,
                'scope': scope_name,
                'metric': m,
                'pearson_r': binned[m]['pearson_r'],
                'pearson_p': binned[m]['pearson_p'],
                'spearman_r': binned[m]['spearman_r'],
                'spearman_p': binned[m]['spearman_p'],
                'bin_dists': binned[m].get('bin_dists', []),
                'bin_perfs': binned[m].get('bin_perfs', []),
            })

    test_df = test_df.copy()
    test_df['distance'] = topk_combined
    return rows, test_df


# ============================================================================
# Visualization
# ============================================================================

def plot_cv_degradation_grid(results_df, n_folds, output_dir):
    """Plot AUCROC degradation curves per fold."""
    fold_indices = sorted(results_df['fold'].unique())
    n_plots = len(fold_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    scope_colors = {'seen_heavy': '#2ecc71', 'unseen_heavy': '#e74c3c',
                    'combined': '#3498db'}
    scope_markers = {'seen_heavy': 's', 'unseen_heavy': 'o', 'combined': '^'}
    scope_labels = {'seen_heavy': 'seen-Heavy', 'unseen_heavy': 'unseen-Heavy',
                    'combined': 'combined'}
    available_scopes = sorted(results_df['scope'].unique())

    for i, fold_idx in enumerate(fold_indices):
        ax = axes[i]
        ax.set_title(f'Fold {fold_idx}', fontsize=11, fontweight='bold')

        for scope in available_scopes:
            row = results_df[
                (results_df['fold'] == fold_idx) &
                (results_df['method'] == 'topk') &
                (results_df['scope'] == scope) &
                (results_df['metric'] == 'aucroc')]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            bd = row['bin_dists']
            bp = row['bin_perfs']
            if not bd or not bp:
                continue
            r_val = row['pearson_r']
            p_val = row['pearson_p']
            sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
            pos_flag = ' [!POS]' if r_val > 0 else ''
            lbl = scope_labels.get(scope, scope)
            ax.plot(bd, bp, f'{scope_markers.get(scope, "o")}-',
                    color=scope_colors.get(scope, '#333'),
                    label=f'{lbl} r={r_val:+.3f}{sig}{pos_flag}',
                    markersize=4, linewidth=1.2)

        ax.set_xlabel('Distance', fontsize=9)
        ax.set_ylabel('AUCROC', fontsize=9)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)

    plt.suptitle('BCR Binding Ab-Stratified 5-Fold CV: AUCROC Degradation\n'
                 'sigma_C + weighted_max_znorm (antibody-stratified, all variants seen)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'cv_degradation_grid.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_cv_summary_bars(results_df, n_folds, output_dir):
    """Bar chart of AUCROC Pearson r per fold."""
    available_scopes = sorted(results_df['scope'].unique())
    scope_colors = {'seen_heavy': '#2ecc71', 'unseen_heavy': '#e74c3c',
                    'combined': '#3498db'}
    scope_labels = {'seen_heavy': 'Seen-Heavy', 'unseen_heavy': 'Unseen-Heavy',
                    'combined': 'Combined'}

    n_panels = len(available_scopes)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fold_indices = sorted(results_df['fold'].unique())
    for ax, scope in zip(axes, available_scopes):
        x = np.arange(len(fold_indices))
        rs = []
        for f in fold_indices:
            row = results_df[
                (results_df['fold'] == f) &
                (results_df['method'] == 'topk') &
                (results_df['scope'] == scope) &
                (results_df['metric'] == 'aucroc')]
            rs.append(row.iloc[0]['pearson_r'] if len(row) > 0 else np.nan)

        colors = ['#e74c3c' if (not np.isnan(r) and r > 0)
                  else scope_colors.get(scope, '#3498db') for r in rs]
        ax.bar(x, rs, color=colors, edgecolor='black', linewidth=0.3, width=0.6)

        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel('Pearson r', fontsize=11)
        ax.set_title(scope_labels.get(scope, scope), fontsize=13,
                     fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'fold{f}' for f in fold_indices], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.suptitle('BCR Binding Ab-Stratified AUCROC Pearson r per Fold\n'
                 'Red bars = positive r (wrong direction)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'cv_summary_bars.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_seen_unseen_heavy(results_df, n_folds, output_dir):
    """2x5 grid: seen-Heavy (top) vs unseen-Heavy (bottom) per fold."""
    fold_indices = sorted(results_df['fold'].unique())
    n_plots = len(fold_indices)

    fig, axes = plt.subplots(2, n_plots, figsize=(5 * n_plots, 8))
    if n_plots == 1:
        axes = axes.reshape(2, 1)

    scope_colors = {'seen_heavy': '#2ecc71', 'unseen_heavy': '#e74c3c'}
    scope_labels = {'seen_heavy': 'Seen-Heavy', 'unseen_heavy': 'Unseen-Heavy'}

    for row_idx, scope in enumerate(['seen_heavy', 'unseen_heavy']):
        for col_idx, fold_idx in enumerate(fold_indices):
            ax = axes[row_idx, col_idx]
            ax.set_title(f'Fold {fold_idx} — {scope_labels[scope]}',
                         fontsize=10, fontweight='bold')

            row = results_df[
                (results_df['fold'] == fold_idx) &
                (results_df['method'] == 'topk') &
                (results_df['scope'] == scope) &
                (results_df['metric'] == 'aucroc')]
            if len(row) == 0:
                ax.text(0.5, 0.5, 'No data',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='gray', style='italic')
                continue
            row = row.iloc[0]
            bd = row['bin_dists']
            bp = row['bin_perfs']
            if not bd or not bp:
                ax.text(0.5, 0.5, 'No data',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='gray', style='italic')
                continue
            r_val = row['pearson_r']
            p_val = row['pearson_p']
            sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
            pos_flag = ' [!POS]' if r_val > 0 else ''
            ax.plot(bd, bp, 'o-',
                    color=scope_colors[scope],
                    label=f'r={r_val:+.3f}{sig}{pos_flag}',
                    markersize=5, linewidth=1.5)

            ax.set_xlabel('Distance', fontsize=9)
            ax.set_ylabel('AUCROC', fontsize=9)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)

    plt.suptitle('BCR Binding Ab-Stratified: Seen-Heavy vs Unseen-Heavy\n'
                 'AUCROC Degradation (sigma_C + weighted_max_znorm)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'seen_unseen_heavy_auroc.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_seen_unseen_heavy_overlay(results_df, n_folds, output_dir):
    """Overlay: seen-Heavy vs unseen-Heavy curves, all folds overlaid."""
    fold_indices = sorted(results_df['fold'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scope_colors_map = {'seen_heavy': '#2ecc71', 'unseen_heavy': '#e74c3c'}
    scope_labels = {'seen_heavy': 'Seen-Heavy', 'unseen_heavy': 'Unseen-Heavy'}

    fold_alphas = np.linspace(0.5, 1.0, len(fold_indices))
    fold_markers = ['o', 's', '^', 'D', 'v'][:len(fold_indices)]

    for ax_idx, scope in enumerate(['seen_heavy', 'unseen_heavy']):
        ax = axes[ax_idx]
        has_data = False
        for f_i, fold_idx in enumerate(fold_indices):
            row = results_df[
                (results_df['fold'] == fold_idx) &
                (results_df['method'] == 'topk') &
                (results_df['scope'] == scope) &
                (results_df['metric'] == 'aucroc')]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            bd = row['bin_dists']
            bp = row['bin_perfs']
            if not bd or not bp:
                continue
            has_data = True
            r_val = row['pearson_r']
            p_val = row['pearson_p']
            sig = '*' if (not np.isnan(p_val) and p_val < 0.05) else ''
            pos_flag = ' [!]' if r_val > 0 else ''
            ax.plot(bd, bp, f'{fold_markers[f_i]}-',
                    color=scope_colors_map[scope],
                    alpha=fold_alphas[f_i],
                    label=f'fold{fold_idx} r={r_val:+.3f}{sig}{pos_flag}',
                    markersize=5, linewidth=1.3)

        ax.set_title(scope_labels[scope], fontsize=12, fontweight='bold')
        ax.set_xlabel('Distance', fontsize=10)
        ax.set_ylabel('AUCROC', fontsize=10)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
        ax.grid(True, alpha=0.3, linestyle='--')
        if has_data:
            ax.legend(fontsize=8, loc='best')
        else:
            ax.text(0.5, 0.5, 'No data',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray', style='italic')

    plt.suptitle('BCR Binding Ab-Stratified: Seen-Heavy vs Unseen-Heavy Overlay\n'
                 'AUCROC Degradation (sigma_C + weighted_max_znorm, all folds overlaid)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'seen_unseen_heavy_overlay.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_seen_unseen_heavy_summary(results_df, output_dir):
    """Save per-fold seen-Heavy vs unseen-Heavy correlation summary."""
    rows = []
    for scope in ['seen_heavy', 'unseen_heavy', 'combined']:
        sub = results_df[
            (results_df['method'] == 'topk') &
            (results_df['scope'] == scope)]
        if len(sub) == 0:
            continue
        for _, r in sub.iterrows():
            rows.append({
                'fold': r['fold'],
                'scope': r['scope'],
                'metric': r['metric'],
                'pearson_r': r['pearson_r'],
                'pearson_p': r['pearson_p'],
                'spearman_r': r['spearman_r'],
                'spearman_p': r['spearman_p'],
            })

    summary_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'seen_unseen_heavy_summary.csv')
    summary_df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ============================================================================
# Summary printing
# ============================================================================

def print_cv_summary(results_df, n_folds, eval_metrics):
    """Print summary comparison table."""
    fold_indices = sorted(results_df['fold'].unique())

    print(f"\n{'='*100}")
    print(f"5-FOLD CV SUMMARY: BCR Binding (XBCR-net, antibody-stratified)")
    print(f"{'='*100}")

    for metric in ['aucroc', 'ap', 'f1']:
        if metric not in eval_metrics:
            continue

        print(f"\n  {metric.upper()} Pearson r (negative = proper degradation):")
        print(f"  {'Scope':>14}", end='')
        for f in fold_indices:
            print(f"  {'fold' + str(f):>10}", end='')
        print(f"  {'Mean':>10}  {'Std':>10}  {'#pos':>5}")
        print(f"  {'-'*94}")

        for scope in ['seen_heavy', 'unseen_heavy', 'combined']:
            sub = results_df[
                (results_df['method'] == 'topk') &
                (results_df['scope'] == scope) &
                (results_df['metric'] == metric)]
            if len(sub) == 0:
                continue

            rs = []
            line = f"  {scope:>14}"
            for f in fold_indices:
                row = sub[sub['fold'] == f]
                if len(row) > 0:
                    r = row.iloc[0]['pearson_r']
                    rs.append(r)
                    flag = ' [!]' if r > 0 else ''
                    line += f"  {r:>7.4f}{flag:>3}"
                else:
                    line += f"  {'N/A':>10}"

            valid_rs = [r for r in rs if not np.isnan(r)]
            n_pos = sum(1 for r in valid_rs if r > 0)
            mean_r = np.mean(valid_rs) if valid_rs else np.nan
            std_r = np.std(valid_rs) if len(valid_rs) > 1 else 0.0
            line += f"  {mean_r:>10.4f}  {std_r:>10.4f}  {n_pos:>5}"
            print(line)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BCR Binding — Antibody-Stratified CV with XBCR-net')
    parser.add_argument('--xbcrnet-dir', type=str,
                        default='Model/XBCR-net',
                        help='Path to XBCR-net directory')
    parser.add_argument('--output-dir', type=str,
                        default='results/xbcr/bind_ab_cv_evaluation',
                        help='Output directory')
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--k', type=float, default=0.1)
    parser.add_argument('--b', type=float, default=0.03)
    parser.add_argument('--bin-num', type=int, default=8)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--eval-metrics', nargs='+',
                        default=['aucroc', 'ap', 'acc', 'f1', 'prec', 'recall'])
    parser.add_argument('--chain-cols', nargs='+',
                        default=['Heavy', 'Light'],
                        help='Chain columns for LogDist')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip XBCR-net training (use existing predictions)')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Do not restore pretrained binding model')
    parser.add_argument('--data-prefix', type=str, default=None,
                        help='Prefix for XBCR-net data_name (default: '
                             'bind_ab if pretrain, bind_ab_nopt if no-pretrain)')
    parser.add_argument('--folds-to-run', nargs='+', type=int, default=None,
                        help='Run only these fold indices')
    args = parser.parse_args()

    t_start = time.time()

    xbcrnet_dir = os.path.join(PROJECT_ROOT, args.xbcrnet_dir) \
        if not os.path.isabs(args.xbcrnet_dir) else args.xbcrnet_dir
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir) \
        if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    chain_cols = args.chain_cols
    restore_pretrain = 0 if args.no_pretrain else 1

    # Data prefix: separate XBCR-net directories for pretrain vs no-pretrain
    if args.data_prefix is not None:
        data_prefix = args.data_prefix
    elif args.no_pretrain:
        data_prefix = 'bind_ab_nopt'
    else:
        data_prefix = 'bind_ab'

    fold_indices = args.folds_to_run if args.folds_to_run is not None \
        else list(range(args.n_folds))

    print(f"{'='*80}")
    print(f"BCR Binding — Antibody-Stratified CV with XBCR-net")
    print(f"{'='*80}")
    print(f"  XBCR-net: {xbcrnet_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Params: K={args.K}, k={args.k}, b={args.b}, bins={args.bin_num}")
    print(f"  Folds: {args.n_folds} total, running: {fold_indices}")
    print(f"  Epochs: {args.max_epochs}")
    print(f"  Chains: {' + '.join(chain_cols)}")
    print(f"  Pretrain: {'yes' if restore_pretrain else 'no'}")
    print(f"  Data prefix: {data_prefix}")
    print(f"  Skip training: {args.skip_training}")
    _flush()

    # ====================================================================
    # Step 1: Load binding data
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Step 1: Loading binding data")
    print(f"{'='*60}")
    _flush()
    pooled = load_binding_data(PROJECT_ROOT)

    # ====================================================================
    # Step 2: Create antibody-stratified CV splits
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Step 2: Creating antibody-stratified {args.n_folds}-fold splits")
    print(f"{'='*60}")
    _flush()
    ab_assignments, folds = create_antibody_stratified_splits(
        pooled, n_folds=args.n_folds, seed=args.seed)

    # Save assignments
    ab_assignments.to_csv(
        os.path.join(output_dir, 'ab_fold_assignments.csv'), index=False)

    # Save fold train/test CSVs
    for k_fold in fold_indices:
        tr, te = folds[k_fold]
        fold_dir = os.path.join(output_dir, f'fold{k_fold}')
        os.makedirs(fold_dir, exist_ok=True)
        tr.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)

    if not args.skip_training:
        # ====================================================================
        # Step 3: Prepare XBCR-net training data
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"Step 3: Preparing XBCR-net data per fold")
        print(f"{'='*60}")
        _flush()
        for k_fold in fold_indices:
            print(f"\n  --- Fold {k_fold} ---")
            train_df, test_df = folds[k_fold]
            prepare_xbcrnet_data(k_fold, train_df, test_df, xbcrnet_dir,
                                 data_prefix=data_prefix)

        # ====================================================================
        # Step 4: Train XBCR-net per fold
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"Step 4: Training XBCR-net per fold ({args.max_epochs} epochs)")
        print(f"{'='*60}")
        _flush()
        for k_fold in fold_indices:
            t_fold = time.time()
            success = train_xbcrnet_fold(
                k_fold, xbcrnet_dir, args.max_epochs,
                restore_pretrain=restore_pretrain,
                data_prefix=data_prefix)
            elapsed = time.time() - t_fold
            status = 'OK' if success else 'FAILED'
            print(f"  Fold {k_fold}: {status} ({elapsed:.1f}s)")
            _flush()

        # ====================================================================
        # Step 5: Run inference per fold
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"Step 5: Running XBCR-net inference per fold")
        print(f"{'='*60}")
        _flush()

    # ====================================================================
    # Step 6: Collect predictions
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Step 6: Collecting predictions")
    print(f"{'='*60}")
    _flush()

    folds_with_preds = []
    for k_fold in fold_indices:
        train_df, test_df = folds[k_fold]

        # Check existing predictions
        existing_test = os.path.join(output_dir, f'fold{k_fold}', 'test.csv')
        if args.skip_training and os.path.exists(existing_test):
            test_with_pred = pd.read_csv(existing_test)
            if 'pred_prob' in test_with_pred.columns:
                # Ensure heavy_seen column exists
                if 'heavy_seen' not in test_with_pred.columns:
                    train_heavies = set(train_df['Heavy'].unique())
                    test_with_pred['heavy_seen'] = test_with_pred['Heavy'].apply(
                        lambda h: 1 if h in train_heavies else 0)
                print(f"  Fold {k_fold}: loaded existing predictions "
                      f"({len(test_with_pred)} rows)")
                folds_with_preds.append((k_fold, train_df, test_with_pred))
                continue

        # Try XBCR-net results (CSV or xlsx)
        data_name = f'{data_prefix}_fold{k_fold}'
        result_base = os.path.join(
            xbcrnet_dir, 'data', data_name, 'test', 'results',
            f'results_rbd_XBCR_net-{k_fold}')
        result_path = result_base + '.csv' if os.path.exists(result_base + '.csv') \
            else result_base + '.xlsx'
        if os.path.exists(result_path):
            pred_df = pd.read_csv(result_path) if result_path.endswith('.csv') \
                else pd.read_excel(result_path)
            print(f"  Fold {k_fold}: loaded XBCR-net output "
                  f"({len(pred_df)} rows)")
        else:
            # Run inference
            model_path = os.path.join(
                xbcrnet_dir, 'models', data_name,
                f'{data_name}-XBCR_net',
                f'model_rbd_{k_fold}.tf.index')
            if not os.path.exists(model_path):
                print(f"  ERROR: no model or predictions for fold {k_fold}")
                continue
            prepare_xbcrnet_data(k_fold, train_df, test_df, xbcrnet_dir,
                                 data_prefix=data_prefix)
            pred_df = infer_xbcrnet_fold(k_fold, xbcrnet_dir,
                                          data_prefix=data_prefix)

        if pred_df is None:
            print(f"  Skipping fold {k_fold}: inference failed")
            continue

        test_with_pred = collect_predictions(k_fold, pred_df, test_df, train_df)

        fold_dir = os.path.join(output_dir, f'fold{k_fold}')
        os.makedirs(fold_dir, exist_ok=True)
        test_with_pred.to_csv(
            os.path.join(fold_dir, 'test.csv'), index=False)

        folds_with_preds.append((k_fold, train_df, test_with_pred))

    if not folds_with_preds:
        print("\nERROR: No folds with predictions available. Exiting.")
        sys.exit(1)

    n_actual_folds = len(folds_with_preds)
    print(f"\n  Collected predictions for {n_actual_folds}/{args.n_folds} folds")
    _flush()

    # Quick AUROC check
    print(f"\n  Overall AUROC per fold:")
    for k_fold, train_df, test_df in folds_with_preds:
        auroc = safe_metric('aucroc', test_df['rbd'].values,
                            test_df['pred_prob'].values)
        n_heavy_seen = (test_df['heavy_seen'] == 1).sum()
        n_heavy_unseen = (test_df['heavy_seen'] == 0).sum()
        print(f"    Fold {k_fold}: AUROC={auroc:.4f}, N={len(test_df)}, "
              f"heavy_seen={n_heavy_seen}, heavy_unseen={n_heavy_unseen}")

    # ====================================================================
    # Step 7: LogDist degradation analysis
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"Step 7: LogDist degradation analysis")
    print(f"{'='*60}")
    _flush()

    all_rows = []
    for k_fold, train_df, test_df in folds_with_preds:
        t_eval = time.time()
        rows, test_with_dist = evaluate_fold_degradation(
            k_fold, train_df, test_df, chain_cols,
            args.k, args.b, args.K, args.bin_num,
            args.eval_metrics, seed=args.seed)
        all_rows.extend(rows)

        # Save test.csv with distance column
        fold_dir = os.path.join(output_dir, f'fold{k_fold}')
        test_with_dist.to_csv(
            os.path.join(fold_dir, 'test.csv'), index=False)

        print(f"  Fold {k_fold} evaluated in {time.time() - t_eval:.1f}s")
        _flush()

    results_df = pd.DataFrame(all_rows)

    # Print summary
    print_cv_summary(results_df, n_actual_folds, args.eval_metrics)

    # Save CSV
    save_df = results_df.drop(columns=['bin_dists', 'bin_perfs'])
    save_df.to_csv(os.path.join(output_dir, 'cv_evaluation_summary.csv'),
                   index=False)

    # Plots
    print(f"\n  Generating CV plots...")
    plot_cv_degradation_grid(results_df, n_actual_folds, output_dir)
    plot_cv_summary_bars(results_df, n_actual_folds, output_dir)
    plot_seen_unseen_heavy(results_df, n_actual_folds, output_dir)
    plot_seen_unseen_heavy_overlay(results_df, n_actual_folds, output_dir)
    save_seen_unseen_heavy_summary(results_df, output_dir)

    # ====================================================================
    # Final summary
    # ====================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'='*80}")

    aucroc_combined = results_df[
        (results_df['method'] == 'topk') &
        (results_df['metric'] == 'aucroc') &
        (results_df['scope'] == 'combined')]
    rs = aucroc_combined['pearson_r'].dropna()
    if len(rs) > 0:
        n_pos = (rs > 0).sum()
        print(f"\n  AUCROC combined: mean r={rs.mean():.4f} "
              f"(std={rs.std():.4f}), #pos={n_pos}/{len(rs)}")

    for scope in ['seen_heavy', 'unseen_heavy']:
        sub = results_df[
            (results_df['method'] == 'topk') &
            (results_df['metric'] == 'aucroc') &
            (results_df['scope'] == scope)]
        rs_s = sub['pearson_r'].dropna()
        if len(rs_s) > 0:
            n_pos_s = (rs_s > 0).sum()
            print(f"  AUCROC {scope}: mean r={rs_s.mean():.4f} "
                  f"(std={rs_s.std():.4f}), #pos={n_pos_s}/{len(rs_s)}")

    print(f"\n  Output: {output_dir}")
    print(f"  Files:")
    print(f"    ab_fold_assignments.csv")
    print(f"    cv_evaluation_summary.csv")
    print(f"    cv_degradation_grid.png")
    print(f"    cv_summary_bars.png")
    print(f"    seen_unseen_heavy_auroc.png")
    print(f"    seen_unseen_heavy_overlay.png")
    print(f"    seen_unseen_heavy_summary.csv")
    for k_fold, _, _ in folds_with_preds:
        print(f"    fold{k_fold}/train.csv, fold{k_fold}/test.csv")


if __name__ == '__main__':
    main()

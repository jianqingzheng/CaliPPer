#!/usr/bin/env python3
"""BCR Binding Cross-Test with fold4 test as calibration set.

Design (3 invariants that MUST hold):
    1. SAME MODEL: Train ONCE on fold4_train → predict fold4_test (cal) AND
       all external test sets (A1-A11, unseen, flu) from the SAME model weights.
    2. SAME DISTANCE: ALL distances (cal + test) computed with sigma_C 3-chain
       + weighted_max_znorm from fold4_train. NEVER use pre-existing CV distances
       (those are 2-chain sigma_H — incompatible).
    3. SAME TRAINING: Overlap removal uses fold4_train (not full 17,903 pool).

Why this design:
    - fold99 (trained on ALL data) has no clean held-out cal data
    - fold4 test provides 3,655 cal samples vs A1-A11↔unseen LOO (221-1091)
    - fold4 train (14,273 unique triplets) leaves MORE clean external samples
      than fold99 (17,903): A1-A11 281 vs 221, unseen 1256 vs 1091

⚠ NEVER:
    - Use CV fold4/test.csv pred_prob or distance columns (wrong model/metric)
    - Mix predictions from different model instances in cal vs test
    - Use fold99 crosstest predictions (different training data)

Usage:
    python eval_bcr_bind_ct_fold4cal.py --models xbcr deepaai mambaaai mint rleaai
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from General_Eval.combine_first_helpers import (
    compute_chain_weights, compute_combine_first_distances)
from General_Eval.general_evaluator import safe_metric
from General_Eval.s2dd import (predict_metric, predict_subset_metric,
                                fit_recalibration, apply_recalibration)
from scipy.stats import pearsonr

# ── Config ──────────────────────────────────────────────────────────────
BCR_CHAINS = ['Heavy', 'Light', 'variant_seq']
BCR_k, BCR_b, BCR_K = 0.1, 0.03, 30
METRICS = ['aucroc', 'ap']
FOLD = 4

CV_BASE = 'results/xbcr/combined_bind_ab_cv'
OUTPUT_DIR = 'results/bcr_bind_ct_fold4cal'

TEST_SET_PATHS = {
    'A1-A11': 'Data/bcr_seq/metabcr_bind_0611/A1-A11_testset',
    'unseen': 'Data/bcr_seq/metabcr_bind_0611/unseen_testset',
    'flu':    'Data/bcr_seq/flu_bind/clean_test',
}

ALL_MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
MODEL_DISPLAY = {'xbcr': 'XBCR-net', 'deepaai': 'DeepAAI', 'mambaaai': 'MambaAAI',
                  'mint': 'MINT', 'rleaai': 'RLEAAI'}


def load_fold_data(fold_idx):
    """Load fold training and test data (sequences + labels only, NO pred/dist)."""
    train = pd.read_csv(os.path.join(CV_BASE, f'fold{fold_idx}', 'train.csv'))
    test = pd.read_csv(os.path.join(CV_BASE, f'fold{fold_idx}', 'test.csv'))
    # Strip pred_prob and distance — must be recomputed per model
    for col in ['pred_prob', 'distance']:
        if col in test.columns:
            test = test.drop(columns=[col])
    print(f"Fold {fold_idx}: train={len(train)}, test={len(test)} (pred/dist stripped)")
    return train, test


def load_external_test_set(ts_name, ts_dir):
    """Load an external binding test set."""
    import glob
    files = glob.glob(os.path.join(ts_dir, '*.xlsx'))
    if files:
        files = [f for f in files if 'result' not in os.path.basename(f).lower()]
        if files:
            df = pd.read_excel(files[0])
        else:
            return None
    else:
        files = glob.glob(os.path.join(ts_dir, '*.csv'))
        if files:
            files = [f for f in files if 'result' not in os.path.basename(f).lower()]
            if files:
                df = pd.read_csv(files[0])
            else:
                return None
        else:
            return None

    required = ['Heavy', 'Light', 'variant_seq', 'rbd']
    for c in required:
        if c not in df.columns:
            return None
    df = df[required].copy()
    df = df.dropna(subset=['Heavy', 'Light', 'variant_seq']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['Heavy', 'Light', 'variant_seq']).reset_index(drop=True)
    df['rbd'] = df['rbd'].astype(int)
    return df


def remove_overlap(test_df, train_df, ts_name):
    """Remove training-test overlap at triplet level."""
    train_keys = set(zip(train_df['Heavy'], train_df['Light'], train_df['variant_seq']))
    mask = ~test_df.apply(
        lambda r: (r['Heavy'], r['Light'], r['variant_seq']) in train_keys, axis=1)
    clean = test_df[mask].reset_index(drop=True)
    print(f"  {ts_name}: {len(test_df)} → {len(clean)} clean ({len(test_df)-len(clean)} overlap)")
    return clean


def compute_distances(df, train_df, cache_path=None):
    """Compute S2DD distances: sigma_C 3-chain + weighted_max_znorm from train_df."""
    if cache_path and os.path.exists(cache_path):
        dists = pd.read_csv(cache_path)['distance'].values
        if len(dists) == len(df):
            print(f"      Loaded cached ({len(dists)})")
            return dists
        print(f"      Cache size mismatch ({len(dists)} vs {len(df)}), recomputing")
    weights, _ = compute_chain_weights(
        train_df, BCR_CHAINS, BCR_k, BCR_b, BCR_K, formula='sigma_C')
    dists = compute_combine_first_distances(
        df, train_df, BCR_CHAINS, weights, BCR_k, BCR_b, BCR_K)
    if cache_path:
        pd.DataFrame({'distance': dists}).to_csv(cache_path, index=False)
    return dists


def run_evaluation(cal_df, test_sets):
    """Run performance prediction and recalibration using model-specific cal."""
    cal_y = cal_df['rbd'].values.astype(int)
    cal_p = cal_df['pred_prob'].values.astype(float)
    cal_d = cal_df['distance'].values.astype(float)
    cal_data = {'fold_test': (cal_y, cal_p, cal_d)}

    print(f"    Cal: n={len(cal_df)}, prev={cal_y.mean():.3f}, "
          f"d=[{cal_d.min():.2f},{cal_d.max():.2f}]")

    ppv_p, npv_p, pp, pn, *_ = fit_recalibration(cal_data)

    for ts_name, ts_df in test_sets.items():
        test_y = ts_df['rbd'].values.astype(int)
        test_p = ts_df['pred_prob'].values.astype(float)
        test_d = ts_df['distance'].values.astype(float)

        for metric in METRICS:
            result = predict_metric(cal_data, test_p, test_d, metrics=[metric])
            actual = safe_metric(metric, test_y, test_p)
            err = abs(result['estimated'][metric] - actual)
            print(f"    {ts_name} {metric}: actual={actual:.3f}, "
                  f"pred={result['estimated'][metric]:.3f}, err={err:.3f}")

        orig_auc = safe_metric('aucroc', test_y, test_p)
        cal_s = apply_recalibration(test_y, test_p, test_d, ppv_p, npv_p, pp, pn)
        recal_auc = safe_metric('aucroc', test_y, cal_s)
        print(f"    {ts_name} recal: {orig_auc:.3f}→{recal_auc:.3f} "
              f"Δ={recal_auc-orig_auc:+.3f}")


# ── Model runners ────────────────────────────────────────────────────
# Each returns (cal_preds, {ts_name: test_preds}) from ONE model instance.

def _run_xbcr(train_df, cal_test_df, ext_test_sets, args):
    """XBCR-net: use saved fold4 model weights for all predictions."""
    import subprocess

    xbcrnet_dir = os.path.join(os.path.dirname(__file__), 'Model', 'XBCR-net')
    # Use fold index 95 + combined_bind_nopt prefix (matching CV data format)
    # fold94 was trained with wrong data prep (SARS-only format, no variant_seq in neg)
    model_fold_idx = 95
    data_prefix = 'combined_bind_nopt'
    data_name = f'{data_prefix}_fold{model_fold_idx}'

    model_check = os.path.join(xbcrnet_dir, 'models', data_name,
                                f'{data_name}-XBCR_net',
                                f'model_rbd_{model_fold_idx}.tf.index')
    from eval_bcr_bind_ab_stratified import (
        prepare_xbcrnet_data, train_xbcrnet_fold, infer_xbcrnet_fold,
        collect_predictions)

    # Train if model doesn't exist
    if not os.path.exists(model_check):
        print(f"    Training XBCR-net (fold{model_fold_idx}, no pretrain)...")
        train_df_x = train_df.copy()
        train_df_x['not_rbd'] = (1 - train_df_x['rbd']).astype(int)
        # Use first test set to prepare data (will be overwritten for each inference)
        dummy_test = cal_test_df.copy()
        dummy_test['not_rbd'] = (1 - dummy_test['rbd']).astype(int)
        prepare_xbcrnet_data(model_fold_idx, train_df_x, dummy_test, xbcrnet_dir,
                              data_prefix=data_prefix)
        success = train_xbcrnet_fold(model_fold_idx, xbcrnet_dir, max_epochs=100,
                                      restore_pretrain=0, data_prefix=data_prefix)
        if not success:
            print("    Training FAILED")
            return None, {}
        print("    Training complete. Model saved.")
        # Sanity check: model must be better than random on cal set
        print("    Sanity check on cal set...")
        _dummy_test = cal_test_df.copy()
        _dummy_test['not_rbd'] = (1 - _dummy_test['rbd']).astype(int)
        prepare_xbcrnet_data(model_fold_idx, train_df_x, _dummy_test, xbcrnet_dir,
                              data_prefix=data_prefix)
        _pred = infer_xbcrnet_fold(model_fold_idx, xbcrnet_dir, data_prefix=data_prefix)
        if _pred is not None:
            _preds = collect_predictions(model_fold_idx, _pred, _dummy_test, train_df_x)
            if _preds is not None:
                _auc = safe_metric('aucroc', _dummy_test['rbd'].values.astype(int),
                                    _preds['pred_prob'].values.astype(float))
                print(f"    Cal AUROC = {_auc:.3f}")
                if _auc < 0.55:
                    print(f"    ⚠ AUROC < 0.55 — model may be broken! STOPPING.")
                    return None, {}

    def _infer(df, label):
        """Run inference using saved model. Prepares test data then infers."""
        df_x = df.copy()
        df_x['not_rbd'] = (1 - df_x['rbd']).astype(int)
        train_df_x = train_df.copy()
        train_df_x['not_rbd'] = (1 - train_df_x['rbd']).astype(int)

        prepare_xbcrnet_data(model_fold_idx, train_df_x, df_x, xbcrnet_dir,
                              data_prefix=data_prefix)
        pred_df = infer_xbcrnet_fold(model_fold_idx, xbcrnet_dir,
                                      data_prefix=data_prefix)
        if pred_df is None:
            print(f"      {label}: inference FAILED")
            return None
        preds = collect_predictions(model_fold_idx, pred_df, df_x, train_df_x)
        if preds is None:
            return None
        matched = preds['pred_prob'].notna().sum()
        print(f"      {label}: {matched}/{len(df)} matched")
        return preds['pred_prob'].values

    # Predict cal (fold4 test) and all external sets with SAME model
    print("    Predicting cal (fold4 test)...")
    cal_preds = _infer(cal_test_df, 'cal')

    ext_preds = {}
    for ts_name, ts_df in ext_test_sets.items():
        print(f"    Predicting {ts_name}...")
        ext_preds[ts_name] = _infer(ts_df, ts_name)

    return cal_preds, ext_preds


def _run_pytorch(model_name, train_df, cal_test_df, ext_test_sets, args):
    """Train a PyTorch model ONCE, predict cal + all test sets from same weights.
    Saves model weights to disk for reuse."""
    import torch

    if model_name == 'rleaai':
        rleaai_path = os.path.join(os.path.dirname(__file__), 'Model', 'RLEAAI')
        if rleaai_path not in sys.path:
            sys.path.insert(0, rleaai_path)

    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, 'model_weights.pt')

    # Concatenate cal + all test sets for single-pass prediction
    all_test = pd.concat([cal_test_df] + list(ext_test_sets.values()),
                          ignore_index=True)

    if model_name == 'deepaai':
        from eval_deepaai_combined_ab_stratified import (
            preprocess_fold, train_deepaai, predict_deepaai)
        features = preprocess_fold(train_df, all_test)
        if os.path.exists(weights_path):
            print(f"      Loading saved weights: {weights_path}")
            model_state = torch.load(weights_path, map_location='cpu')
        else:
            model_state, history = train_deepaai(features, epochs=args.deepaai_epochs)
            print(f"      val_AUC: {history.get('best_val_auc', 'N/A')}")
            torch.save(model_state, weights_path)
            print(f"      Saved weights: {weights_path}")
        all_preds = predict_deepaai(features, model_state)

    elif model_name == 'mambaaai':
        from eval_mambaaai_combined_ab_stratified import (
            preprocess_fold, train_mambaaai, predict_mambaaai)
        features = preprocess_fold(train_df, all_test)
        if os.path.exists(weights_path):
            print(f"      Loading saved weights: {weights_path}")
            model_state = torch.load(weights_path, map_location='cpu')
        else:
            model_state, history = train_mambaaai(features, epochs=args.mamba_epochs)
            print(f"      val_AUC: {history.get('best_val_auc', 'N/A')}")
            torch.save(model_state, weights_path)
            print(f"      Saved weights: {weights_path}")
        all_preds = predict_mambaaai(features, model_state)

    elif model_name == 'mint':
        import json, argparse as _ap
        from eval_mint_combined_ab_stratified import train_mint_fold
        cfg_path = os.path.join(os.path.dirname(__file__), 'Model', 'MINT', 'data',
                                 'esm2_t33_650M_UR50D.json')
        ckpt = os.path.join(os.path.dirname(__file__), 'Model', 'MINT', 'mint.ckpt')
        if not os.path.exists(cfg_path) or not os.path.exists(ckpt):
            return None, {}
        with open(cfg_path) as f:
            cfg = _ap.Namespace(**json.load(f))
        # MINT uses sklearn MLP — no torch weights, but we can cache predictions
        all_preds = train_mint_fold(train_df, all_test, cfg, ckpt,
                                     device=args.device, seed=42)

    elif model_name == 'rleaai':
        from eval_rleaai_combined_ab_stratified import (
            preprocess_fold, train_rleaai, predict_rleaai)
        features = preprocess_fold(train_df, all_test, device=args.device)
        if os.path.exists(weights_path):
            print(f"      Loading saved weights: {weights_path}")
            model_state = torch.load(weights_path, map_location='cpu')
        else:
            model_state, history = train_rleaai(features, epochs=args.rleaai_epochs)
            print(f"      val_AUC: {history.get('best_val_auc', 'N/A')}")
            torch.save(model_state, weights_path)
            print(f"      Saved weights: {weights_path}")
        all_preds = predict_rleaai(features, model_state)
    else:
        return None, {}

    if all_preds is None:
        return None, {}

    # Split predictions back to cal + each test set
    offset = 0
    cal_preds = all_preds[offset:offset + len(cal_test_df)]
    offset += len(cal_test_df)

    ext_preds = {}
    for ts_name, ts_df in ext_test_sets.items():
        ext_preds[ts_name] = all_preds[offset:offset + len(ts_df)]
        offset += len(ts_df)

    print(f"      cal: {len(cal_preds)}, " +
          ", ".join(f"{k}: {len(v)}" for k, v in ext_preds.items()))
    return cal_preds, ext_preds


def main():
    parser = argparse.ArgumentParser(
        description='BCR CT with fold4 test as calibration')
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--models', nargs='+', default=['xbcr'],
                        choices=ALL_MODELS)
    parser.add_argument('--fold', type=int, default=FOLD)
    parser.add_argument('--distances-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--deepaai-epochs', type=int, default=200)
    parser.add_argument('--mamba-epochs', type=int, default=50)
    parser.add_argument('--rleaai-epochs', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load fold data (NO pred/dist — recomputed per model) ──────
    print(f"Loading fold {args.fold} data...")
    train_df, cal_test_df = load_fold_data(args.fold)

    # ── Load + clean external test sets ───────────────────────────
    print("\nLoading external test sets...")
    ext_test_sets = {}
    for ts_name, ts_dir in TEST_SET_PATHS.items():
        df = load_external_test_set(ts_name, ts_dir)
        if df is not None:
            clean = remove_overlap(df, train_df, ts_name)
            if len(clean) >= 50:
                ext_test_sets[ts_name] = clean

    # ── Compute S2DD distances (sigma_C 3-chain, ALL from fold train) ─
    print(f"\nComputing S2DD distances (sigma_C 3-chain from fold{args.fold} train)...")

    # Cal distances
    cal_dist_path = os.path.join(args.output_dir, f'cal_fold{args.fold}_distances.csv')
    t0 = time.time()
    cal_dists = compute_distances(cal_test_df, train_df, cache_path=cal_dist_path)
    cal_test_df = cal_test_df.copy()
    cal_test_df['distance'] = cal_dists
    print(f"    cal: {len(cal_dists)} distances ({time.time()-t0:.1f}s)")

    # Test distances
    for ts_name, ts_df in ext_test_sets.items():
        dist_path = os.path.join(args.output_dir,
                                  f'{ts_name}_fold{args.fold}_distances.csv')
        t0 = time.time()
        dists = compute_distances(ts_df, train_df, cache_path=dist_path)
        ext_test_sets[ts_name] = ts_df.copy()
        ext_test_sets[ts_name]['distance'] = dists
        print(f"    {ts_name}: {len(dists)} distances ({time.time()-t0:.1f}s)")

    if args.distances_only:
        print("\n--distances-only: done")
        return

    # ── Per-model: train once → predict cal + all test → evaluate ─
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  {MODEL_DISPLAY[model_name]} (fold{args.fold} model)")
        print(f"{'='*60}")

        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Check if all outputs exist
        cal_path = os.path.join(model_dir, 'cal_predictions.csv')
        all_exist = os.path.exists(cal_path) and all(
            os.path.exists(os.path.join(model_dir, f'{ts}_predictions.csv'))
            for ts in ext_test_sets)

        if all_exist and not args.evaluate_only:
            print("  All predictions exist, loading...")

        if not all_exist and not args.evaluate_only:
            # Train + predict
            print("  Training and predicting (single model instance)...")
            if model_name == 'xbcr':
                cal_preds, ext_preds = _run_xbcr(
                    train_df, cal_test_df, ext_test_sets, args)
            else:
                cal_preds, ext_preds = _run_pytorch(
                    model_name, train_df, cal_test_df, ext_test_sets, args)

            if cal_preds is None:
                print("  FAILED")
                continue

            # Save cal predictions
            cal_out = cal_test_df.copy()
            cal_out['pred_prob'] = cal_preds
            cal_out.to_csv(cal_path, index=False)

            # Save test predictions
            for ts_name, preds in ext_preds.items():
                if preds is not None:
                    out_df = ext_test_sets[ts_name].copy()
                    out_df['pred_prob'] = preds
                    out_df.to_csv(os.path.join(model_dir,
                                                f'{ts_name}_predictions.csv'),
                                   index=False)
                    y = out_df['rbd'].values.astype(int)
                    auc = safe_metric('aucroc', y, preds)
                    print(f"    {ts_name}: AUROC={auc:.3f}")

        # ── Evaluate with model-specific cal ──────────────────────
        print(f"\n  Evaluating with model-specific cal...")
        if os.path.exists(cal_path):
            model_cal = pd.read_csv(cal_path)
        else:
            print("  No cal predictions, skipping")
            continue

        loaded_tests = {}
        for ts_name in ext_test_sets:
            pred_path = os.path.join(model_dir, f'{ts_name}_predictions.csv')
            if os.path.exists(pred_path):
                loaded_tests[ts_name] = pd.read_csv(pred_path)

        if loaded_tests:
            run_evaluation(model_cal, loaded_tests)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for model_name in args.models:
        model_dir = os.path.join(args.output_dir, model_name)
        cal_path = os.path.join(model_dir, 'cal_predictions.csv')
        if not os.path.exists(cal_path):
            continue
        model_cal = pd.read_csv(cal_path)
        cal_y = model_cal['rbd'].values.astype(int)
        cal_p = model_cal['pred_prob'].values.astype(float)
        cal_d = model_cal['distance'].values.astype(float)
        cal_data = {'fold_test': (cal_y, cal_p, cal_d)}
        ppv_p, npv_p, pp, pn, *_ = fit_recalibration(cal_data)

        print(f"\n  {MODEL_DISPLAY[model_name]}:")
        for ts_name in ext_test_sets:
            pred_path = os.path.join(model_dir, f'{ts_name}_predictions.csv')
            if not os.path.exists(pred_path):
                continue
            df = pd.read_csv(pred_path)
            y = df['rbd'].values.astype(int)
            p = df['pred_prob'].values.astype(float)
            d = df['distance'].values.astype(float)
            auc = safe_metric('aucroc', y, p)
            cal_s = apply_recalibration(y, p, d, ppv_p, npv_p, pp, pn)
            recal_auc = safe_metric('aucroc', y, cal_s)
            r = predict_metric(cal_data, p, d, metrics=['aucroc'])
            err = abs(r['estimated']['aucroc'] - auc)
            print(f"    {ts_name:8s}: AUROC={auc:.3f}, "
                  f"pred_err={err:.3f}, recal Δ={recal_auc-auc:+.3f}")


if __name__ == '__main__':
    main()

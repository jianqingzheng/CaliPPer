"""Derive the 12 BCR Fig4 cache CSVs from the committed audit baseline CSV.

Background: the original `results/bcr_bind_ct_fold4cal/{model}/*_predictions.csv`
files (the input to `cache_bcr_fig4_fold4cal.py`) were lost in the 2026-05-20
filter-repo destruction. The committed
`Manuscript/designed_figures/panels/fig4/audit_bcr_baseline_results.csv`
preserves the per-(model, split, subset, strategy, metric) actual/s2dd/pape/mcbpe
values that were used to produce the published panels — this script maps that
audit CSV into the 12-file cache schema that `gen_fig4_unique_scatter.py` and
`gen_fig4_heatmaps_reordered.py` consume.

Output schema (matches the TCR cache produced by `cache_tcr_fig4_v27_predictions.py`):
    subset, metric, predicted, actual, n, prevalence, source, seen, model, mean_dist

Where:
    predicted    = audit.s2dd  (S2DD method prediction; PAPE/MCBPE used elsewhere)
    actual       = audit.actual
    n            = audit.n
    prevalence   = computed from today's pooled cal+external test data per vhash
                   (antigen strategy) or overall pool mean (distance strategy)
    source       = audit.strategy  ('antigen' or 'distance')
    seen         = 'all'  (BCR doesn't have train-epitope-seen flag like TCR)
    model        = audit.model
    mean_dist    = NaN  (BCR panels don't use this column for plotting)

Outputs 12 files in `results/fig3_fig4_bcr_cache/`:
    bcr_fig4_fold4cal_{split}_{strategy}_{metric}.csv
    where split  in {ct, cv}
          strategy in {antigen, distance}
          metric in {aucroc, ap, f1}
"""

import hashlib
import os
import sys

import pandas as pd

INPUT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)

AUDIT_PATH = os.path.join(
    INPUT_DIR, 'Manuscript', 'designed_figures', 'panels', 'fig4',
    'audit_bcr_baseline_results.csv')
CACHE_DIR = os.path.join(INPUT_DIR, 'results', 'fig3_fig4_bcr_cache')


def vhash(s):
    """MD5[:12] hash of variant sequence — matches audit 'subset' for antigen strategy."""
    return hashlib.md5(str(s).encode()).hexdigest()[:12]


def compute_pool_stats(project_root):
    """Pool today's cal (fold4 test) + external test sets → per-vhash stats.

    Returns:
        antigen_prev: dict[vhash → prevalence]
        antigen_domain: dict[vhash → 'sars'|'flu'] (inferred domain that
            originally passed the >=30 + cal-class-balance filter; matches
            the LOO record the audit kept)
        overall_prev: float (for distance bins)
    """
    cal_path = os.path.join(
        project_root, 'results', 'xbcr', 'combined_bind_ab_cv',
        'fold4', 'test.csv')
    if not os.path.exists(cal_path):
        raise FileNotFoundError(
            f"Missing {cal_path}. Run "
            f"`python eval_bcr_combined_ab_stratified.py --skip-training "
            f"--folds-to-run 4` first.")
    cal = pd.read_csv(cal_path)

    sys.path.insert(0, project_root)
    from eval_bcr_bind_ct_fold4cal import (
        load_external_test_set, TEST_SET_PATHS)
    ext_dfs = []
    for ts_name, ts_dir in TEST_SET_PATHS.items():
        df = load_external_test_set(ts_name, ts_dir)
        if df is not None:
            ext_dfs.append(df)

    pooled = pd.concat([cal] + ext_dfs, ignore_index=True)
    pooled['vhash'] = pooled['variant_seq'].apply(vhash)
    pooled['ds'] = pooled.get('data_source', pd.Series(['sars']*len(pooled))).fillna('sars')

    antigen_prev = pooled.groupby('vhash')['rbd'].mean().to_dict()

    # Infer domain per vhash: if >=30 sars samples then 'sars' (dominant
    # in LOO survival); else if >=30 flu, 'flu'; else fall back to majority.
    MIN_SAMPLES = 30
    antigen_domain = {}
    for vh, grp in pooled.groupby('vhash'):
        n_sars = (grp['ds'] == 'sars').sum()
        n_flu = (grp['ds'] == 'flu').sum()
        if n_sars >= MIN_SAMPLES:
            antigen_domain[vh] = 'sars'
        elif n_flu >= MIN_SAMPLES:
            antigen_domain[vh] = 'flu'
        else:
            antigen_domain[vh] = 'sars' if n_sars >= n_flu else 'flu'

    overall_prev = float(pooled['rbd'].mean())
    return antigen_prev, antigen_domain, overall_prev


def main():
    print(f"Reading audit: {AUDIT_PATH}")
    audit = pd.read_csv(AUDIT_PATH)
    print(f"  Loaded {len(audit)} rows, columns: {list(audit.columns)}")

    print(f"\nComputing per-vhash prevalence + domain from today's pooled data...")
    antigen_prev, antigen_domain, overall_prev = compute_pool_stats(INPUT_DIR)
    print(f"  Antigen subsets: {len(antigen_prev)}, overall pool prev: {overall_prev:.3f}")

    print(f"\nDeriving cache rows...")
    cache_rows = []
    for _, row in audit.iterrows():
        is_ct_antigen = (row['split'] == 'CT') and (row['strategy'] == 'antigen')
        if row['strategy'] == 'antigen':
            prev = antigen_prev.get(row['subset'], overall_prev)
        else:
            prev = overall_prev
        # MATCH original cache schema: CT antigen had subset='variant' + source='{domain}_{vhash}'
        # CV antigen + all distance had subset=actual_subset_name + source=test set name
        if is_ct_antigen:
            domain = antigen_domain.get(row['subset'], 'sars')
            subset_val = 'variant'
            source_val = f'{domain}_{row["subset"]}'
        else:
            subset_val = row['subset']
            source_val = row['strategy']
        cache_rows.append({
            'subset': subset_val,
            'metric': row['metric'],
            'predicted': row['s2dd'],
            'actual': row['actual'],
            'n': row['n'],
            'prevalence': prev,
            'source': source_val,
            'seen': 'all',
            'model': row['model'],
            'mean_dist': float('nan'),
        })
    cache_df = pd.DataFrame(cache_rows)

    audit_keys = audit[['split', 'strategy', 'metric']]
    cache_df = cache_df.assign(_split=audit_keys['split'].values,
                                _strategy=audit_keys['strategy'].values)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"\nWriting 12 cache files to {CACHE_DIR}/")
    n_written = 0
    for split in ['ct', 'cv']:
        for strategy in ['antigen', 'distance']:
            for metric in ['aucroc', 'ap', 'f1']:
                sub = cache_df[
                    (cache_df['_split'] == split.upper())
                    & (cache_df['_strategy'] == strategy)
                    & (cache_df['metric'] == metric)
                ].drop(columns=['_split', '_strategy'])
                if len(sub) == 0:
                    print(f"  WARN: 0 rows for {split}/{strategy}/{metric} — skipping")
                    continue
                fname = f'bcr_fig4_fold4cal_{split}_{strategy}_{metric}.csv'
                out_path = os.path.join(CACHE_DIR, fname)
                sub.to_csv(out_path, index=False)
                print(f"  Wrote {fname} ({len(sub)} rows)")
                n_written += 1

    print(f"\nDone. {n_written}/12 cache files written.")


if __name__ == '__main__':
    main()

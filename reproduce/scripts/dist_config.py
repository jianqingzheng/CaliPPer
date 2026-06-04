"""Distance type configuration for panel generation.

Controls which S2DD distance metric is used across all fig2-5 panel scripts.
Set via environment variable: DIST_TYPE=blosum-sqrt (default: lev-log)

Usage in panel scripts:
    from dist_config import DIST_TYPE, get_tcr_dist_path, get_output_dir, get_bcr_ct_distance

Two modes:
  lev-log:      Levenshtein + log(k*(1-ratio+b)) transform (original)
  blosum-sqrt:  BLOSUM-SW + sqrt(1-sim) transform (superior for prediction/recalibration)
"""
import os

# ── Distance type selection ──
DIST_TYPE = os.environ.get('DIST_TYPE', 'lev-log')
assert DIST_TYPE in ('lev-log', 'blosum-sqrt'), \
    f"DIST_TYPE must be 'lev-log' or 'blosum-sqrt', got '{DIST_TYPE}'"

# ── File suffix for .npy distance files ──
DIST_SUFFIX = {
    'lev-log': '_dist.npy',
    'blosum-sqrt': '_blosumsqrt_dist.npy',
}

# For TCR epitope strategy (uniform combining)
DIST_SUFFIX_UNIFORM = {
    'lev-log': '_uniform_dist.npy',
    'blosum-sqrt': '_blosumsqrt_uniform_dist.npy',
}

# ── Output subfolder ──
DIST_SUBDIR = {
    'lev-log': 'lev-logtransf',
    'blosum-sqrt': 'blosum-sqrt',
}

# ── BCR CT distance source ──
# lev-log: read 'distance' column from fold4cal CSVs
# blosum-sqrt: read sidecar .npy files alongside CSVs
BCR_DIST_MODE = {
    'lev-log': 'csv_column',
    'blosum-sqrt': 'npy_sidecar',
}

# ── Display label ──
DIST_LABEL = {
    'lev-log': 'S2DD distance (Levenshtein)',
    'blosum-sqrt': 'S2DD distance (BLOSUM-sqrt)',
}


# ── Helper functions ──

def get_tcr_dist_path(cache_dir, model, split, name, strategy='degradation'):
    """Get .npy distance path for TCR data.

    Args:
        cache_dir: base cache directory (e.g., results/fig2_cache)
        model: model name (e.g., 'nettcr')
        split: 'ct' or 'cv'
        name: test set name (e.g., 'seen_test') or fold (e.g., 'fold0')
        strategy: 'degradation' (sigma_C) or 'epitope' (uniform)
    """
    if strategy == 'epitope':
        suffix = DIST_SUFFIX_UNIFORM[DIST_TYPE]
    else:
        suffix = DIST_SUFFIX[DIST_TYPE]

    if split == 'ct':
        return os.path.join(cache_dir, f'{model}_ct_{name}{suffix}')
    else:
        # CV: try _combined variant first for lev-log
        if DIST_TYPE == 'lev-log' and strategy != 'epitope':
            combined = os.path.join(cache_dir, f'{model}_cv_{name}_combined{suffix}')
            if os.path.exists(combined):
                return combined
        return os.path.join(cache_dir, f'{model}_cv_{name}{suffix}')


def get_bcr_ct_distance(df, model_dir, test_set_name):
    """Get BCR CT distances — from CSV column (lev) or .npy sidecar (blosum).

    Args:
        df: DataFrame with 'distance' column (used for lev-log)
        model_dir: path to model's fold4cal directory
        test_set_name: e.g., 'cal_predictions', 'A1-A11', 'unseen', 'flu'

    Returns:
        numpy array of distances (same length as df)
    """
    import numpy as np
    if BCR_DIST_MODE[DIST_TYPE] == 'csv_column':
        return df['distance'].values.astype(float)
    else:
        npy_path = os.path.join(model_dir,
                                f'{test_set_name}_blosumsqrt_dist.npy')
        if os.path.exists(npy_path):
            return np.load(npy_path).astype(float)
        else:
            raise FileNotFoundError(
                f"BLOSUM-sqrt sidecar not found: {npy_path}\n"
                f"Run precompute_blosum_sqrt_distances.py first.")


def get_output_dir(base_panel_dir):
    """Get output directory for panels, creating subdirectory if needed.

    Returns:
        path to output directory (e.g., fig2/blosum-sqrt/)
    """
    subdir = DIST_SUBDIR[DIST_TYPE]
    out = os.path.join(base_panel_dir, subdir)
    os.makedirs(out, exist_ok=True)
    return out


# Print config on import
if os.environ.get('DIST_TYPE'):
    print(f"[dist_config] DIST_TYPE={DIST_TYPE}, suffix={DIST_SUFFIX[DIST_TYPE]}, "
          f"output={DIST_SUBDIR[DIST_TYPE]}")

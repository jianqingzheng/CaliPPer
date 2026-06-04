"""Pluggable distance metric for S2DD — accepts arbitrary per-chain similarity function.

This module provides a new entry point that parallels `compute_combine_first_distances`
in `combine_first_helpers.py` but allows the per-chain similarity function to be swapped.

The existing Levenshtein-based pipeline in `combine_first_helpers.py` is UNCHANGED.
Use this module only when you need a different per-chain distance (e.g., BLOSUM-SW,
ESM-2 embedding, k-mer).

Example:
    import parasail
    def sw_sim(q, r):
        q_self = parasail.sw_stats(q, q, 10, 1, parasail.blosum62).score
        r_self = parasail.sw_stats(r, r, 10, 1, parasail.blosum62).score
        s = parasail.sw_stats(q, r, 10, 1, parasail.blosum62).score
        return s / np.sqrt(q_self * r_self) if q_self > 0 and r_self > 0 else 0.0

    dists = compute_s2dd_pluggable(
        test_df, train_df, chain_cols=['peptide', 'CDR3b'],
        weights=np.array([0.9997, 0.0003]),
        similarity_fn=sw_sim,
        k=0.1, b=0.1, K=50,
    )
"""
import numpy as np
import time


def compute_pairwise_similarity_matrix(qry_seqs, ref_seqs, similarity_fn,
                                        verbose=True, progress_every=200):
    """Compute pairwise similarity matrix: (n_qry_unique, n_ref_unique).

    Args:
        qry_seqs: iterable of query sequences (will be deduplicated)
        ref_seqs: iterable of reference sequences (will be deduplicated)
        similarity_fn: callable(q_str, r_str) -> float in [0, 1]
        verbose: print progress
        progress_every: print every N query sequences

    Returns:
        sim: (n_qry_unique, n_ref_unique) float32 array
        unique_qry: list of unique query sequences (order matches sim rows)
        unique_ref: list of unique ref sequences (order matches sim cols)
    """
    unique_qry = list(dict.fromkeys(qry_seqs))
    unique_ref = list(dict.fromkeys(ref_seqs))

    if verbose:
        print(f"  Computing pairwise similarity: {len(unique_qry)} × {len(unique_ref)} "
              f"= {len(unique_qry) * len(unique_ref):,}")
    t0 = time.time()

    sim = np.zeros((len(unique_qry), len(unique_ref)), dtype=np.float32)
    for i, q in enumerate(unique_qry):
        for j, r in enumerate(unique_ref):
            sim[i, j] = similarity_fn(q, r)
        if verbose and ((i + 1) % progress_every == 0 or i == len(unique_qry) - 1):
            elapsed = time.time() - t0
            print(f"    [{i+1:>5}/{len(unique_qry)}] {(i+1)/len(unique_qry)*100:.1f}% "
                  f"({elapsed:.0f}s)")

    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")
    return sim, unique_qry, unique_ref


def compute_s2dd_pluggable(test_df, train_df, chain_cols, weights,
                            similarity_fn, k=0.1, b=0.1, K=50,
                            chain_stats_subsample=50, verbose=True,
                            cache_prefix=None, cache_dir=None,
                            transform='log', return_stats=False):
    """S2DD distances using a pluggable per-chain similarity function.

    Mirrors the logic of `compute_combine_first_distances` but with swappable
    per-chain similarity. The existing Levenshtein pipeline is unaffected.

    Formula per chain (depends on `transform`):
        transform='log':  d(q, r) = log(k * (1 - sim(q, r) + b))
        transform='sqrt': d(q, r) = sqrt(max(1 - sim(q, r), 0))
                          (no k/b needed; variance-stabilizing for [0,1] proportions)
        d_chain(q) = mean of top-K smallest d(q, r) over training rows
    Combine (weighted_max_znorm style):
        combined(q) = sum_c w_c * (d_chain_c(q) - mu_c) / sigma_c

    Args:
        test_df, train_df: dataframes with chain columns
        chain_cols: list of column names (one per chain)
        weights: chain weights (e.g., sigma_C weights from Levenshtein pipeline)
        similarity_fn: callable(str, str) -> float in [0, 1]
        k, b: log-distance parameters (default 0.1, 0.1; unused when transform='sqrt')
        K: top-K neighbors (default 50)
        chain_stats_subsample: N for train-vs-train mu/sigma estimation
        cache_prefix, cache_dir: if provided, cache per-chain sim matrices
        transform: 'log' (default, backward compat) or 'sqrt' (optimal for BLOSUM-SW)

    Returns:
        numpy array of per-test-row distances (shape: n_test)
    """
    import warnings
    from pathlib import Path
    n_chains = len(chain_cols)
    n_test = len(test_df)

    # Check sequence lengths vs transform and warn if mismatch
    for col in chain_cols:
        sample = test_df[col].astype(str).head(100)
        mean_len = sample.str.len().mean()
        if transform == 'sqrt' and mean_len > 100:
            warnings.warn(
                f"BLOSUM-sqrt on long sequences ({col}: mean {mean_len:.0f} AA). "
                f"BLOSUM-SW similarity saturates for sequences >100 AA with few "
                f"mutations, compressing distances and losing discrimination. "
                f"Consider Levenshtein-log (transform='log') instead. "
                f"See feedback_blosum_vs_lev_limitation.md for evidence.",
                UserWarning, stacklevel=2)
        elif transform == 'log' and mean_len <= 20 and similarity_fn is not None:
            # Only warn if a BLOSUM similarity_fn was provided but log transform chosen
            pass  # Levenshtein-log is the default, no warning needed

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    chain_qry_dists = []
    chain_stats = []

    for ch_idx, col in enumerate(chain_cols):
        qry_seqs = test_df[col].astype(str).tolist()
        ref_seqs = train_df[col].astype(str).tolist()

        # Compute or load cached similarity matrix
        cache_path = None
        if cache_prefix and cache_dir is not None:
            cache_path = cache_dir / f'{cache_prefix}_{col}.npz'

        cache_valid = False
        if cache_path and cache_path.exists():
            if verbose:
                print(f"  Loading cached {col} similarity from {cache_path}")
            npz = np.load(cache_path, allow_pickle=True)
            sim = npz['sim']
            unique_qry = npz['qry'].tolist()
            unique_ref = npz['ref'].tolist()
            # Verify cache covers all current query and ref sequences
            qry_set = set(unique_qry)
            ref_set = set(unique_ref)
            missing_qry = sorted(set(qry_seqs) - qry_set)
            missing_ref = sorted(set(ref_seqs) - ref_set)
            if missing_ref:
                if verbose:
                    print(f"    Cache missing {len(missing_ref)} ref seqs — full recompute")
            elif missing_qry:
                # Partial cache hit: compute only missing query rows
                if verbose:
                    print(f"    Cache partial: {len(missing_qry)} missing qry — computing delta")
                delta_sim, _, _ = compute_pairwise_similarity_matrix(
                    missing_qry, unique_ref, similarity_fn, verbose=verbose)
                # Merge: append missing rows to sim matrix
                sim = np.vstack([sim, delta_sim])
                unique_qry = unique_qry + missing_qry
                # Update cache with expanded matrix
                if cache_path:
                    np.savez_compressed(cache_path, sim=sim,
                                         qry=np.array(unique_qry),
                                         ref=np.array(unique_ref))
                cache_valid = True
            else:
                cache_valid = True
        if not cache_valid:
            if verbose:
                print(f"  Chain {col}: computing pairwise similarity")
            sim, unique_qry, unique_ref = compute_pairwise_similarity_matrix(
                qry_seqs, ref_seqs, similarity_fn, verbose=verbose)
            if cache_path:
                np.savez_compressed(cache_path, sim=sim,
                                     qry=np.array(unique_qry),
                                     ref=np.array(unique_ref))

        # Build index maps
        qry_idx_map = {s: i for i, s in enumerate(unique_qry)}
        ref_idx_map = {s: i for i, s in enumerate(unique_ref)}
        qry_row_idx = np.array([qry_idx_map[s] for s in qry_seqs])
        ref_row_idx = np.array([ref_idx_map[s] for s in ref_seqs])

        # Convert similarity → distance
        if transform == 'sqrt':
            dist_u = np.sqrt(np.maximum(1.0 - sim, 0.0))
        elif transform == 'raw':
            dist_u = 1.0 - sim
        else:  # 'log' (default)
            dist_u = np.log(k * (1.0 - sim + b))

        # Expand to (n_test, n_train_rows) then take topK mean
        all_dists = dist_u[qry_row_idx[:, None], ref_row_idx[None, :]]
        topk = np.sort(all_dists, axis=1)[:, :K].mean(axis=1)
        chain_qry_dists.append(topk)

        # Z-norm mu/sigma from TRAIN-vs-TRAIN (fixed reference frame).
        # This ensures all inputs (cal, test, different batches) are on the
        # same z-norm scale. Using input-sample mu/sigma would give different
        # scales for cal vs test, breaking PPV/NPV curve transfer.
        # The train anchor (raw distance = 0) is transformed with the SAME
        # mu/sigma but is NOT included in the mu/sigma computation.
        rng = np.random.RandomState(42)
        n_unique_ref = len(unique_ref)
        n_sub = min(chain_stats_subsample, n_unique_ref)
        sub_idx = rng.choice(n_unique_ref, size=n_sub, replace=False)
        sub_seqs = [unique_ref[i] for i in sub_idx]

        tt_sim = np.zeros((n_sub, n_unique_ref), dtype=np.float32)
        for ii, q in enumerate(sub_seqs):
            for jj, r in enumerate(unique_ref):
                tt_sim[ii, jj] = similarity_fn(q, r)
        if transform == 'sqrt':
            tt_dist_u = np.sqrt(np.maximum(1.0 - tt_sim, 0.0))
        elif transform == 'raw':
            tt_dist_u = 1.0 - tt_sim
        else:
            tt_dist_u = np.log(k * (1.0 - tt_sim + b))
        tt_all = tt_dist_u[:, ref_row_idx]
        tt_topk = np.sort(tt_all, axis=1)[:, :K].mean(axis=1)
        mu = float(tt_topk.mean())
        sigma = float(tt_topk.std()) if tt_topk.std() >= 1e-9 else 1.0
        chain_stats.append((mu, sigma))

        if verbose:
            print(f"    {col}: mu={mu:.4f}, sigma={sigma:.4f}")

    # Auto-compute sigma_C weights from internal sigma if weights='auto'
    if weights is None or (isinstance(weights, str) and weights == 'auto'):
        from collections import Counter
        auto_weights = np.zeros(n_chains)
        for c, col in enumerate(chain_cols):
            seqs = train_df[col].astype(str).tolist()
            freq = Counter(seqs)
            n_seqs = len(seqs)
            simpson_c = sum(f * (f - 1) for f in freq.values()) / (n_seqs * (n_seqs - 1)) if n_seqs > 1 else 1.0
            _, sigma_c = chain_stats[c]  # use internally-computed sigma (full-row)
            auto_weights[c] = sigma_c * simpson_c
        auto_weights = auto_weights / auto_weights.sum() if auto_weights.sum() > 0 else np.ones(n_chains) / n_chains
        weights = auto_weights
        if verbose:
            print(f"  Auto sigma_C weights: {', '.join(f'{col}={w:.4f}' for col, w in zip(chain_cols, weights))}")

    # Combine: weighted sum of z-normalized per-chain distances
    combined = np.zeros(n_test)
    for c in range(n_chains):
        mu_c, sigma_c = chain_stats[c]
        combined += weights[c] * (chain_qry_dists[c] - mu_c) / sigma_c

    if return_stats:
        return combined, chain_stats
    return combined


def make_sw_blosum62_similarity(gap_open=10, gap_extend=1):
    """Factory for BLOSUM62 normalized Smith-Waterman similarity.

    Returns callable(q, r) -> float. Caches self-scores within the closure
    for efficiency across many queries/references.
    """
    import parasail
    self_cache = {}

    def _self_score(s):
        if s not in self_cache:
            self_cache[s] = parasail.sw_stats(s, s, gap_open, gap_extend,
                                               parasail.blosum62).score
        return self_cache[s]

    def sw_sim(q, r):
        qs = _self_score(q)
        rs = _self_score(r)
        if qs <= 0 or rs <= 0:
            return 0.0
        s = parasail.sw_stats(q, r, gap_open, gap_extend, parasail.blosum62).score
        return s / np.sqrt(qs * rs)

    return sw_sim


def make_levenshtein_similarity():
    """Factory for Levenshtein ratio similarity (for direct comparison).

    Note: existing `compute_combine_first_distances` already uses Levenshtein;
    this factory is provided for API consistency / testing the pluggable design.
    """
    import Levenshtein
    def lev_sim(q, r):
        return Levenshtein.ratio(q, r)
    return lev_sim

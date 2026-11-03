"""
Step 2: Cluster TCRs by edit distance and assign hierarchical folds.

Groups similar TCRs (combined alpha+beta edit distance < 4, same V genes,
same epitope) into clusters using connected components. Then assigns each
cluster to one of 5 folds, ensuring:
  - 10x epitopes are restricted to folds 0-2
  - Other epitopes are spread across 4 of 5 folds
  - Fold sizes are balanced

Input:
  - <input_dir>/01_merged_paired_tcrs.tsv

Output:
  - <output_dir>/02_tcrs_with_folds.tsv
"""

import argparse
from collections import defaultdict
from pathlib import Path

import igraph
import numpy as np
import pandas as pd
import polars as pl
from pyrepseq import nn


def compute_close_neighbours(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Find TCR pairs with combined alpha+beta edit distance < 4."""
    cdr3_as = df["cdr3_a_aa"].tolist()
    cdr3_bs = df["cdr3_b_aa"].tolist()

    print(f"  Computing alpha nearest neighbours ({len(cdr3_as):,} sequences)...")
    alpha_neighbours = nn.nearest_neighbor(cdr3_as, max_edits=3, n_cpu=4)
    print(f"  Computing beta nearest neighbours ({len(cdr3_bs):,} sequences)...")
    beta_neighbours = nn.nearest_neighbor(cdr3_bs, max_edits=3, n_cpu=4)

    combined_distances: dict[tuple[int, int], int] = {}

    for a_idx1, a_idx2, a_dist in alpha_neighbours:
        if df.loc[a_idx1, "v_a_gene"] == df.loc[a_idx2, "v_a_gene"]:
            combined_distances[(a_idx1, a_idx2)] = a_dist

    for b_idx1, b_idx2, b_dist in beta_neighbours:
        if df.loc[b_idx1, "v_b_gene"] == df.loc[b_idx2, "v_b_gene"]:
            key = (b_idx1, b_idx2)
            if key in combined_distances:
                combined_distances[key] += b_dist
            else:
                combined_distances[key] = b_dist

    close_neighbours = [
        (idx1, idx2)
        for (idx1, idx2), dist in combined_distances.items()
        if dist < 4
        and df.loc[idx1, "v_a_gene"] == df.loc[idx2, "v_a_gene"]
        and df.loc[idx1, "v_b_gene"] == df.loc[idx2, "v_b_gene"]
        and df.loc[idx1, "epitope"] == df.loc[idx2, "epitope"]
    ]

    return sorted(close_neighbours, key=lambda x: combined_distances[x])


def assign_clusters(df: pd.DataFrame, edges: list[tuple[int, int]]) -> pd.DataFrame:
    """Assign cluster IDs via connected components on the neighbour graph."""
    edge_df = pd.DataFrame(edges, columns=["source", "target"])
    g = igraph.Graph.DataFrame(edge_df, directed=False, vertices=df)
    components = g.connected_components(mode="weak")
    df["cluster"] = components.membership
    return df


def hierarchical_fold_assignment(
    df: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Assign folds hierarchically by epitope and cluster.

    Matches notebook 3 (notebooks/3_sf_full_paired_dataset.ipynb cell
    `hierarchical_fold_assignment`):
      - 10x epitopes restricted to folds 0-2; other epitopes spread across
        4 of 5 folds.
      - Cluster placement: entire clusters go into the same fold (preserves
        leakage protection — same TCR cluster can't be in train and val/test).
      - Rebalance: moves SINGLE ROW INDICES between folds (not whole
        clusters), matching notebook semantics. This breaks cluster
        integrity at the rebalance step but matches what JQ received.
    """
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    grouped = df.groupby(["epitope", "cluster"])

    # fold_assignments[f] is the list of ROW INDICES assigned to fold f.
    # Matches notebook structure exactly (despite naming as cluster_to_move
    # in the rebalance loop, the value is a single row index).
    fold_assignments: dict[int, list[int]] = defaultdict(list)
    epitope_folds: dict[str, np.ndarray] = {}

    for (epitope, _), group in grouped:
        if epitope not in epitope_folds:
            if "10x UMI Denoise" in group["database"].values:
                epitope_folds[epitope] = rng.choice([0, 1, 2], size=3, replace=False)
            else:
                epitope_folds[epitope] = rng.choice(range(n_folds), size=4, replace=False)

        fold = min(epitope_folds[epitope], key=lambda f: len(fold_assignments[f]))
        fold_assignments[fold].extend(group.index)

    # Make sure every fold has an entry even if nothing was assigned to it.
    for f in range(n_folds):
        _ = fold_assignments[f]

    counts = {f: len(fold_assignments[f]) for f in fold_assignments}
    max_iters = 100_000  # generous cap; row-level moves converge fast in practice
    i = 0
    tolerance = len(df) // (n_folds * 2)
    while max(counts.values()) - min(counts.values()) > tolerance and i < max_iters:
        max_fold = max(counts, key=lambda f: counts[f])
        min_fold = min(counts, key=lambda f: counts[f])
        if not fold_assignments[max_fold]:
            break
        # Single row index moves (matches notebook semantics).
        row_to_move = int(rng.choice(fold_assignments[max_fold]))
        fold_assignments[max_fold].remove(row_to_move)
        fold_assignments[min_fold].append(row_to_move)
        counts = {f: len(fold_assignments[f]) for f in fold_assignments}
        i += 1

    if i == max_iters:
        spread = max(counts.values()) - min(counts.values())
        print(
            f"  WARNING: fold rebalance saturated at {max_iters} iters "
            f"with spread={spread} > tolerance={tolerance}"
        )

    df["fold"] = -1
    for fold, indices in fold_assignments.items():
        df.loc[indices, "fold"] = fold

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Cluster and assign folds")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input_dir / "01_merged_paired_tcrs.tsv"
    df = pl.read_csv(input_path, separator="\t")
    print(f"Loaded {df.height:,} rows from {input_path}")

    # Filter to rows with valid CDR3 and V/J genes for both chains
    required = ["cdr3_a_aa", "v_a_gene", "j_a_gene", "cdr3_b_aa", "v_b_gene", "j_b_gene"]
    df = df.filter(pl.all_horizontal(pl.col(required).is_not_null()))
    # Fold assignment uses .sample(frac=1, random_state=seed) which permutes
    # the INPUT order. Upstream Polars groupby/unique do not preserve order,
    # so the TSV we read in can arrive in a different row order from run to
    # run. Sort on a stable key so fold assignments reproduce across runs.
    df = df.sort("cdr3_a_aa", "cdr3_b_aa", "epitope", "database")
    print(f"After null filter: {df.height:,} rows")

    print("Computing TCR neighbour graph...")
    pdf = df.to_pandas()
    close_neighbours = compute_close_neighbours(pdf)
    print(f"  Found {len(close_neighbours):,} close neighbour pairs")

    print("Assigning clusters via connected components...")
    pdf = assign_clusters(pdf, close_neighbours)
    n_clusters = pdf["cluster"].nunique()
    print(f"  Assigned {n_clusters:,} clusters")

    print("Assigning hierarchical folds...")
    pdf = hierarchical_fold_assignment(pdf, seed=args.seed)

    result = pl.DataFrame(pdf)
    print("\nFold distribution:")
    print(result["fold"].value_counts().sort("fold"))

    output_path = args.output_dir / "02_tcrs_with_folds.tsv"
    result.write_csv(output_path, separator="\t")
    print(f"\nWrote {result.height:,} rows to {output_path}")


if __name__ == "__main__":
    main()

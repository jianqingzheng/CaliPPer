"""
Step 0: Denoise 10x Genomics dextramer binding data.

Reproduces `single_10x_binders.csv` from raw 10x CD8+ T-cell CSVs:
  1. Import + merge 4 donor datasets
  2. Normalize dextramer UMIs (subtract max NC, log-normalize)
  3. PCA (8 components) on normalized dextramer space
  4. kNN graph + Leiden clustering
  5. Purity filter: keep clusters binding >92% to a single epitope
  6. Aggregate cells to clonotype level
  7. Filter to clonotypes binding exactly one epitope

Hyperopt random search finds (k, resolution) that maximize NMI between
cluster labels and per-cell max_binder. Results cached to
`<output_dir>/optimal_parameters.json` for reproducibility across runs.

Inputs (relative to project root):
  - resources/external/10x_genomics/10x_processed/cd8+tcellshealthydonor{1..4}_epitopes.csv
  - resources/external/10x_genomics/10x_processed/cd8+tcellshealthydonor{1..4}_tcrs.csv

Output:
  - <output_dir>/single_10x_binders.csv
  - <output_dir>/optimal_parameters.json    (cached hyperopt results)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from tcr_ml.tenx import (
    create_knn_graph,
    identify_pure_clusters,
    import_and_merge_across_donors,
    leiden_clustering,
)

DEXTRAMER_PATTERN = r"^[AB]\d{4}.*$"
NEGATIVE_CONTROL_PATTERN = r".*NC.*"
N_PCA_COMPONENTS = 8
PURITY_THRESHOLD = 0.92
HYPEROPT_N_SAMPLES = 20
K_RANGE = (5, 50)
RESOLUTION_RANGE = (0.1, 2.0)


def _donor_paths(project_root: Path) -> tuple[list[str], list[str]]:
    """Return ordered lists of epitope and TCR CSV paths for 4 donors."""
    base = project_root / "resources/external/10x_genomics/10x_processed"
    epitopes = [str(base / f"cd8+tcellshealthydonor{i}_epitopes.csv") for i in range(1, 5)]
    tcrs = [str(base / f"cd8+tcellshealthydonor{i}_tcrs.csv") for i in range(1, 5)]
    return epitopes, tcrs


def normalize_and_pca(merged_donors: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Normalize dextramer UMIs and run PCA.

    Mirrors notebook 1 cells 3-8: subtract max negative control from
    dextramer UMIs, compute max_binder (argmax epitope), log-normalize,
    PCA to N_PCA_COMPONENTS. Returns (normalised_table, pca_column_names).

    Notebook pre-drops _binder columns in calculate_max_negative_control_and_drop_binders.
    Negative control subtraction uses `max_nc_umi` column computed at import time.
    """
    import re

    dex_cols = [c for c in merged_donors.columns if re.match(DEXTRAMER_PATTERN, c)]
    # Exclude negative controls from dextramer list (NC columns match ^A|B\d{4}.*$ too)
    dex_cols = [c for c in dex_cols if not re.match(NEGATIVE_CONTROL_PATTERN, c)]

    # Step 1: compute per-cell summary + subtract NC + pick max_binder
    with_summary = merged_donors.with_columns(
        pl.sum_horizontal([pl.col(c).fill_null(0) for c in dex_cols]).alias("sum_umis"),
        pl.max_horizontal([pl.col(c).fill_null(0) for c in dex_cols]).alias("max_umis"),
        pl.sum_horizontal([(pl.col(c).fill_null(0) > 0) for c in dex_cols]).alias("n_nonzero"),
        pl.struct([pl.col(c).fill_null(0) for c in dex_cols]).alias("_umis_struct"),
    ).with_columns(
        pl.col("_umis_struct")
        .map_elements(
            lambda x: max(x.items(), key=lambda y: y[1])[0] if max(x.values()) > 0 else "none",
            return_dtype=pl.String,
        )
        .alias("max_binder")
    )

    # Step 2: subtract max negative control from each dextramer column, clip to >= 0
    max_nc = pl.max_horizontal([pl.col(c).fill_null(0) for c in merged_donors.columns if re.match(NEGATIVE_CONTROL_PATTERN, c)])
    subtracted = with_summary.with_columns(
        *[
            pl.max_horizontal(pl.col(c).fill_null(0) - max_nc, pl.lit(0)).alias(c)
            for c in dex_cols
        ]
    )

    # Step 3: log-normalize dextramer cols: (x / row_sum).log1p()
    row_sum = pl.sum_horizontal([pl.col(c) for c in dex_cols])
    normalised = subtracted.with_columns(
        *[
            (pl.col(c) / row_sum).fill_nan(0).log1p().alias(c)
            for c in dex_cols
        ]
    )

    # Step 4: PCA
    X = normalised.select(dex_cols).to_numpy()
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=0)
    pca_out = pca.fit_transform(X)
    pca_cols = [f"PCA{i + 1}" for i in range(N_PCA_COMPONENTS)]
    normalised = pl.concat(
        [normalised, pl.DataFrame(pca_out, schema=pca_cols)],
        how="horizontal",
    )

    return normalised, pca_cols


def run_hyperopt(
    clustering_data: np.ndarray,
    max_binder_values: list[str],
    seed: int,
    n_samples: int = HYPEROPT_N_SAMPLES,
) -> dict:
    """Random-search hyperopt over (k, resolution) maximizing NMI.

    Deterministic given `seed`. Samples parameters via seeded RNG and
    passes the same seed to Leiden for reproducible clustering.
    """
    rng = np.random.RandomState(seed)
    ks = rng.randint(K_RANGE[0], K_RANGE[1] + 1, size=n_samples)
    resolutions = rng.uniform(RESOLUTION_RANGE[0], RESOLUTION_RANGE[1], size=n_samples)

    best = {"best_k": None, "best_resolution": None, "best_score": -float("inf")}
    trials = []

    for i, (k, res) in enumerate(zip(ks.tolist(), resolutions.tolist(), strict=False)):
        k = int(k)
        try:
            graph = create_knn_graph(clustering_data, k)
            labels = leiden_clustering(graph, res, seed=seed).membership
            nmi = normalized_mutual_info_score(labels, max_binder_values)
        except Exception as e:
            print(f"  trial {i + 1}/{n_samples}: k={k}, res={res:.4f} — failed ({e})")
            continue

        trials.append({"k": k, "resolution": res, "nmi": nmi})
        marker = ""
        if nmi > best["best_score"]:
            best = {"best_k": k, "best_resolution": res, "best_score": nmi}
            marker = " *"
        print(f"  trial {i + 1}/{n_samples}: k={k}, res={res:.4f}, nmi={nmi:.4f}{marker}")

    best["all_trials"] = trials
    best["seed"] = seed
    best["n_samples"] = n_samples
    return best


def build_single_binders(
    pure_cells: pl.DataFrame, merged_donors: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate pure-cluster cells to clonotype level, keep single-epitope binders.

    Mirrors notebook 1 cells 60-65: standardise TCR gene nomenclature to
    match the downstream pipeline, group by clonotype, collect unique
    max_binder values, filter to single-epitope clonotypes.
    """
    # Pull TCR sequences back from merged_donors via cell id
    joined = pure_cells.join(
        merged_donors.select(
            "id", "cdr3", "cdr3_nt", "v_gene", "j_gene", "cdr3_b", "cdr3_nt_b", "v_gene_b", "j_gene_b"
        ),
        on="id",
        how="inner",
    ).filter(pl.col("cdr3").is_not_null() & pl.col("cdr3_b").is_not_null())

    standardised = joined.select(
        "id",
        "donor",
        (
            pl.when(pl.col("v_gene").str.contains("D"))
            .then(pl.col("v_gene").str.replace_all(r"(\d+)(D)", r"$1/D"))
            .otherwise(pl.col("v_gene"))
            + pl.lit("*01")
        ).alias("v_a"),
        (pl.col("j_gene") + pl.lit("*01")).alias("j_a"),
        pl.col("cdr3").alias("cdr3_a"),
        pl.col("cdr3_nt").alias("cdr3_a_nt"),
        (pl.col("v_gene_b") + pl.lit("*01")).alias("v_b"),
        (pl.col("j_gene_b") + pl.lit("*01")).alias("j_b"),
        pl.col("cdr3_b"),
        pl.col("cdr3_nt_b").alias("cdr3_b_nt"),
        pl.col("max_binder"),
    )

    clonotype_level = standardised.group_by(
        "cdr3_a", "cdr3_b", "v_a", "j_a", "v_b", "j_b"
    ).agg(
        pl.len().alias("n_cells"),
        pl.col("donor").unique().len().alias("n_donors"),
        pl.col("max_binder").unique().alias("binds_to"),
    )

    single_binders = clonotype_level.filter(
        pl.col("binds_to").list.len() == 1
    ).with_columns(pl.col("binds_to").list.get(0).alias("binds_to"))

    return single_binders


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 0: Denoise 10x dextramer data")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-hyperopt",
        action="store_true",
        help="Use notebook's known-good params (k=20, resolution=0.29267) instead of hyperopt.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params_path = args.output_dir / "optimal_parameters.json"

    print("Loading 10x donors...")
    epitope_paths, tcr_paths = _donor_paths(args.project_root)
    merged_donors = import_and_merge_across_donors(epitope_paths, tcr_paths)
    # Polars full-joins inside aggregate_chains_by_cell do not preserve row
    # order across runs. Sort deterministically by cell id so that PCA input
    # and every downstream kNN / Leiden step reproduces byte-identically.
    merged_donors = merged_donors.sort("id")
    print(f"  Merged donors: {merged_donors.height:,} cells")

    print("Normalizing and running PCA...")
    normalized_with_pca, pca_cols = normalize_and_pca(merged_donors)
    clustering_data = normalized_with_pca.select(pca_cols).to_numpy()
    print(f"  Normalized cells: {normalized_with_pca.height:,}")

    # Decide clustering params: cached → hyperopt → notebook defaults
    if params_path.exists():
        with params_path.open() as f:
            params = json.load(f)
        print(f"Loaded cached hyperopt params from {params_path}")
        k = int(params["best_k"])
        resolution = float(params["best_resolution"])
    elif args.skip_hyperopt:
        k, resolution = 20, 0.29267314185592075
        print(f"Skipping hyperopt; using notebook defaults: k={k}, resolution={resolution}")
    else:
        print(f"Running hyperopt ({HYPEROPT_N_SAMPLES} trials, seed={args.seed})...")
        max_binders = normalized_with_pca["max_binder"].to_list()
        best = run_hyperopt(clustering_data, max_binders, seed=args.seed)
        with params_path.open("w") as f:
            json.dump(best, f, indent=2)
        print(f"  Best: k={best['best_k']}, resolution={best['best_resolution']:.6f}, nmi={best['best_score']:.4f}")
        print(f"  Cached to {params_path}")
        k = int(best["best_k"])
        resolution = float(best["best_resolution"])

    print(f"Running Leiden clustering: k={k}, resolution={resolution:.6f} (seed={args.seed})")
    graph = create_knn_graph(clustering_data, k)
    labels = leiden_clustering(graph, resolution, seed=args.seed).membership

    pure_clusters_info = identify_pure_clusters(
        normalized_with_pca, labels, PURITY_THRESHOLD
    )
    n_pure = pure_clusters_info["optimal_leiden_labels"].n_unique()
    print(f"  Pure clusters (purity > {PURITY_THRESHOLD}): {n_pure}")

    # Filter cells to those in pure clusters (inline — preserves id column)
    norm_with_labels = normalized_with_pca.with_columns(
        pl.Series(labels).alias("optimal_leiden_labels")
    )
    pure_cells = norm_with_labels.join(
        pure_clusters_info, on="optimal_leiden_labels", how="inner"
    )
    print(f"  Pure cells: {pure_cells.height:,}")

    single_binders = build_single_binders(pure_cells, merged_donors)
    # Polars groupby output order is non-deterministic; sort for byte-stable
    # output across runs so CI / reviewer comparisons see no spurious diffs.
    single_binders = single_binders.sort("cdr3_a", "cdr3_b", "v_a", "v_b")
    print(f"  Single-epitope clonotypes: {single_binders.height:,}")

    output_path = args.output_dir / "single_10x_binders.csv"
    single_binders.write_csv(output_path)
    print(f"\nWrote {single_binders.height:,} rows to {output_path}")

    print("\nTop 10 epitopes by clonotype count:")
    print(
        single_binders.group_by("binds_to")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(10)
    )


if __name__ == "__main__":
    main()

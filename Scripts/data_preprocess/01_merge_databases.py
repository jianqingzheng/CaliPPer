"""
Step 1: Merge external TCR databases and internal Dong Lab data.

Loads VDJdb, McPAS, 10x Genomics (UMI-denoised), and Dong Lab paired
COVID TCRs. Filters to paired human TCRs, standardises gene nomenclature,
deduplicates, and validates with pyrepseq.

Inputs (relative to project root):
  - workspace/external/vdjdb/vdjdb_formatted.tsv
  - workspace/external/mcpas/mcpas_formatted.tsv
  - workspace/external/tenx/tenx_formatted.tsv
  - resources/internal/All_single_cell_COVID_paired_TCRs.csv

Output:
  - <output_dir>/01_merged_paired_tcrs.tsv
"""

import argparse
from pathlib import Path

import polars as pl
from pyrepseq import standardize_dataframe

DONGLAB_EPITOPE_MAP: dict[str, str] = {
    "NP16": "SPRWYFYYL",
    "S34": "CTFEYVSQPFLMDLE",
    "S151": "NLLLQYGSFCTQLNR",
    "S174": "TDEMIAQYTSALLAG",
    "M24": "TSRTLSYYKLGASQRVA",
    "ORF3a": "FTSDYYQLY",
}

DONGLAB_MHC_A_MAP: dict[str, str] = {
    # Original map used HLA-DPA1*01:01 and HLA-DRA1*01:01 which are not
    # IMGT-registered (tidytcells nulls them). The actually-registered alleles
    # that exist in mhc_imgt_contacts.tsv are DPA1*01:03 and DRA*01:01.
    "S34": "HLA-DPA1*01:03",
    "S151": "HLA-DRA*01:01",
    "S174": "HLA-DRA*01:01",
    "ORF3a": "HLA-A*02:01",
    "M24": "HLA-DRA*01:01",
    "NP16": "HLA-B*07:02",
}

DONGLAB_MHC_B_MAP: dict[str, str] = {
    "S34": "HLA-DPB1*04:01",
    "S151": "HLA-DRB1*15:01",
    "S174": "HLA-DRB1*15:01",
    "M24": "HLA-DRB1*01:01",
    "ORF3a": "B2M",
    "NP16": "B2M",
}

PYREPSEQ_COL_MAPPER: dict[str, str] = {
    "v_a_gene": "TRAV",
    "j_a_gene": "TRAJ",
    "v_b_gene": "TRBV",
    "j_b_gene": "TRBJ",
    "cdr3_a_aa": "CDR3A",
    "cdr3_b_aa": "CDR3B",
    "epitope": "Epitope",
    "mhc_a": "MHCA",
    "mhc_b": "MHCB",
}


def load_external_dbs(project_root: Path) -> pl.DataFrame:
    """Load and concatenate VDJdb and McPAS."""
    vdjdb = pl.read_csv(
        project_root / "workspace/external/vdjdb/vdjdb_formatted.tsv",
        separator="\t",
    )
    mcpas = pl.read_csv(
        project_root / "workspace/external/mcpas/mcpas_formatted.tsv",
        separator="\t",
    )
    return pl.concat([vdjdb, mcpas], how="diagonal_relaxed")


def load_10x_denoised(project_root: Path, denoised_csv: Path | None = None) -> pl.DataFrame:
    """Load 10x UMI-denoised binders and parse epitope/MHC from binds_to.

    Prefers the single_10x_binders.csv produced by step 00 (denoising
    pipeline) if provided; otherwise falls back to the pre-formatted
    workspace file from the Snakemake pipeline.
    """
    if denoised_csv is not None and denoised_csv.exists():
        src = pl.read_csv(denoised_csv)
    else:
        src = pl.read_csv(
            project_root / "workspace/external/tenx/tenx_formatted.tsv",
            separator="\t",
        )
    # Drop any nucleotide CDR3 columns from step 00 output (not needed downstream)
    drop_cols = [c for c in ("n_cells", "n_donors", "cdr3_a_nt", "cdr3_b_nt") if c in src.columns]
    # binds_to format: "A0301_KLGGALQAK_IE-1_CMV" (4 parts) or
    # "A0201_RMFPNAPYL_WT-1" (3 parts, no species suffix).
    return (
        src
        .drop(drop_cols)
        .rename({"binds_to": "epitope"})
        .with_columns(pl.col("epitope").str.split("_").alias("_parts"))
        .with_columns(
            (
                pl.lit("HLA-")
                + pl.col("_parts").list.get(0).str.slice(0, 1)
                + pl.lit("*")
                + pl.col("_parts").list.get(0).str.slice(1, 2)
                + pl.lit(":")
                + pl.col("_parts").list.get(0).str.slice(3)
            ).alias("mhc_a"),
            pl.lit("B2M").alias("mhc_b"),
            pl.col("_parts").list.get(1, null_on_oob=True).alias("epitope"),
            pl.col("_parts").list.get(2, null_on_oob=True).alias("epitope_gene"),
            pl.col("_parts").list.get(3, null_on_oob=True).alias("epitope_species"),
            pl.lit("HomoSapiens").alias("species"),
            pl.lit("10x_genomics").alias("reference"),
            pl.lit(2).alias("score"),
            pl.lit("dextramer-UMI").alias("method_identification"),
            pl.lit("10x").alias("method_sequencing"),
            pl.lit("yes").alias("is_singlecell"),
            pl.lit("healthy").alias("subject_cohort"),
            pl.lit("PBMC").alias("tissue"),
            pl.lit("10x UMI Denoise").alias("database"),
        )
        .drop("_parts")
    )


def load_donglab(project_root: Path) -> pl.DataFrame:
    """Load Dong Lab COVID paired TCRs with epitope sequence and MHC mapping."""
    return (
        pl.read_csv(
            project_root / "resources/internal/All_single_cell_COVID_paired_TCRs.csv",
            infer_schema_length=500,
        )
        .with_columns(
            pl.lit("HomoSapiens").alias("species"),
            pl.lit("DongLab").alias("reference"),
            pl.lit(3).alias("score"),
            pl.lit("tetramer-sort").alias("method_identification"),
            pl.when(pl.col("cell_name").str.contains(r"^[ACGT]{4}"))
            .then(pl.lit("10x"))
            .otherwise(pl.lit("smartseq2"))
            .alias("method_sequencing"),
            pl.col("cell_name").alias("id"),
            pl.lit("yes").alias("is_singlecell"),
            pl.lit("COVID-19").alias("subject_cohort"),
            pl.lit("PBMC").alias("tissue"),
            pl.when(pl.col("CDR3_alpha").is_not_null() & pl.col("CDR3_beta").is_not_null())
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("is_paired"),
            pl.when(pl.col("CDR3_beta").is_not_null())
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("has_beta"),
            pl.when(pl.col("CDR3_alpha").is_not_null())
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("has_alpha"),
            pl.lit("DongLab").alias("database"),
            (pl.col("TRAV") + pl.lit("*01")).alias("v_a"),
            (pl.col("TRAJ") + pl.lit("*01")).alias("j_a"),
            (pl.col("TRBV") + pl.lit("*01")).alias("v_b"),
            (pl.col("TRBJ") + pl.lit("*01")).alias("j_b"),
            pl.col("epitope").alias("epitope_gene"),
            pl.col("epitope").replace(DONGLAB_EPITOPE_MAP).alias("epitope"),
            pl.col("CDR3_alpha").alias("cdr3_a"),
            pl.col("CDR3_beta").alias("cdr3_b"),
        )
        .drop("TRAV", "TRAJ", "TRBV", "TRBJ", "cell_name", "CDR3_alpha", "CDR3_beta")
        .with_columns(
            pl.col("epitope_gene").replace(DONGLAB_MHC_A_MAP).alias("mhc_a"),
            pl.col("epitope_gene").replace(DONGLAB_MHC_B_MAP).alias("mhc_b"),
        )
        .select(
            "id", "cdr3_a", "v_a", "j_a", "cdr3_b", "v_b", "j_b",
            "epitope", "epitope_gene", "mhc_a", "mhc_b", "species",
            "reference", "score", "method_identification", "method_sequencing",
            "is_singlecell", "subject_cohort", "tissue", "is_paired",
            "has_beta", "has_alpha", "database",
        )
    )


def standardise_gene_names(df: pl.DataFrame) -> pl.DataFrame:
    """Rename to tcrdist convention and ensure allele designations present."""
    rename_map = {
        "cdr3_a": "cdr3_a_aa",
        "cdr3_b": "cdr3_b_aa",
        "v_a": "v_a_gene",
        "j_a": "j_a_gene",
        "v_b": "v_b_gene",
        "j_b": "j_b_gene",
    }
    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    gene_cols = ["v_a_gene", "j_a_gene", "v_b_gene", "j_b_gene"]
    for col in gene_cols:
        if col not in df.columns:
            continue
        df = df.with_columns(
            pl.when(~(pl.col(col).str.contains(r"\*\d{2}$")))
            .then(pl.col(col) + "*01")
            .otherwise(pl.col(col))
            .alias(col),
        )

    # Fix TRDV naming: TRAV12-2D -> TRAV12-2/DV
    if "v_a_gene" in df.columns:
        df = df.with_columns(
            pl.col("v_a_gene").str.replace_all(r"(\d+)(D)", r"$1/D"),
        )

    # Null-fill empty/NA MHC
    df = df.with_columns(
        pl.when(
            pl.col("mhc_a").eq("") | pl.col("mhc_a").is_null() | pl.col("mhc_a").eq("NA")
        )
        .then(pl.lit(None))
        .otherwise(pl.col("mhc_a"))
        .alias("mhc_a"),
    )

    return df


def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    """Add donglab priority column and sort. No row removal.

    The notebook (notebooks/2_generate_collated_dataset.ipynb, cell starting
    `stitchr_db = ...`) deliberately commented out both unique() calls and
    only sorted by a 'donglab' tiebreaker so non-DongLab rows come first.
    The earlier version of this function ran both dropped uniques, which
    silently collapsed cross-database (TCR, epitope) duplicates and lost
    several hundred positives that the notebook preserved.
    """
    df = df.with_columns(
        pl.when(pl.col("database") == "DongLab")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("donglab"),
    )
    return df.sort("donglab", descending=False)


def replace_na_strings(df: pl.DataFrame) -> pl.DataFrame:
    """Replace literal 'NA' sentinel with null in CDR3 and gene columns.

    Uses exact equality, not substring match: real CDR3 sequences frequently
    contain 'NA' as a germline-encoded fragment (e.g. CASSNARLM, CAVRDLLTNAGKSTF),
    and a substring check would silently nuke them.
    """
    target_cols = [
        "cdr3_a_aa", "v_a_gene", "j_a_gene",
        "cdr3_b_aa", "v_b_gene", "j_b_gene",
    ]
    for col in target_cols:
        if col not in df.columns:
            continue
        df = df.with_columns(
            pl.when(pl.col(col).eq("NA"))
            .then(pl.lit(None))
            .otherwise(pl.col(col))
            .alias(col),
        )
    return df.filter(~pl.col("epitope").eq("NA"))


def fill_class_i_b2m(df: pl.DataFrame) -> pl.DataFrame:
    """Restore mhc_b='B2M' for class-I rows where pyrepseq silently nulled it.

    tidytcells rejects the literal 'B2M' at mhc_precision='allele' because
    B2M is the β2-microglobulin gene symbol, not an HLA allele. Since B2M
    is obligate for all class-I TCRs, a null mhc_b on a class-I row is a
    validator side-effect, not missing data. This restores the ground-truth
    without fighting the validator.
    """
    if "mhc_a" not in df.columns or "mhc_b" not in df.columns:
        return df
    class_i_prefixes = ("HLA-A*", "HLA-B*", "HLA-C*", "HLA-E*")
    mhc_a_str = pl.col("mhc_a").cast(pl.Utf8, strict=False)
    is_class_i = mhc_a_str.is_not_null() & pl.any_horizontal(
        [mhc_a_str.str.starts_with(p) for p in class_i_prefixes]
    )
    return df.with_columns(
        pl.when(is_class_i & pl.col("mhc_b").is_null())
        .then(pl.lit("B2M"))
        .otherwise(pl.col("mhc_b"))
        .alias("mhc_b")
    )


def validate_with_pyrepseq(df: pl.DataFrame) -> pl.DataFrame:
    """Run pyrepseq standardize_dataframe for functional TCR validation.

    pyrepseq expects col_mapper as {source: standard}. It renames columns to
    the canonical names (TRAV, CDR3A, ...), IMGT-normalises gene symbols,
    and nulls entries it cannot validate. tcr_precision='allele' preserves
    the *NN suffix required downstream by SCEPTR. Rows with any required
    field nulled out are dropped here.
    """
    pdf = df.to_pandas()
    pdf = standardize_dataframe(
        pdf,
        col_mapper=PYREPSEQ_COL_MAPPER,
        tcr_enforce_functional=True,
        tcr_precision="allele",
        mhc_precision="allele",
        suppress_warnings=True,
    )
    inverse = {v: k for k, v in PYREPSEQ_COL_MAPPER.items()}
    pdf = pdf.rename(columns=inverse)
    required = ["cdr3_a_aa", "v_a_gene", "cdr3_b_aa", "v_b_gene", "epitope"]
    pdf = pdf.dropna(subset=required)
    return pl.DataFrame(pdf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Merge TCR databases")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--denoised-10x-csv",
        type=Path,
        default=None,
        help="Path to single_10x_binders.csv produced by step 00. "
        "If omitted, falls back to workspace/external/tenx/tenx_formatted.tsv.",
    )
    parser.add_argument("--subsample-n", type=int, default=None)
    parser.add_argument("--subsample-seed", type=int, default=42)
    args = parser.parse_args()

    root = args.project_root
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading databases...")
    external = load_external_dbs(root)
    donglab = load_donglab(root)
    print(f"  VDJdb + McPAS: {external.height:,} rows")
    print(f"  DongLab:       {donglab.height:,} rows")

    sources = [external, donglab]
    if args.denoised_10x_csv is not None:
        tenx = load_10x_denoised(root, denoised_csv=args.denoised_10x_csv)
        print(f"  10x denoised:  {tenx.height:,} rows")
        sources.insert(1, tenx)
    else:
        print("  10x denoised:  skipped (--denoised-10x-csv not provided)")

    full_db = pl.concat(sources, how="diagonal_relaxed")
    print(f"  Combined:      {full_db.height:,} rows")

    # Filter to paired human TCRs
    model_db = full_db.filter(
        pl.col("species").eq("HomoSapiens")
        & pl.col("cdr3_a").is_not_null()
        & pl.col("cdr3_b").is_not_null()
    ).select(
        "cdr3_a", "cdr3_b", "v_a", "j_a", "v_b", "j_b",
        "epitope", "epitope_gene", "mhc_a", "mhc_b",
        "reference", "score", "method_identification", "method_sequencing",
        "is_singlecell", "subject_cohort", "tissue", "is_paired", "database",
    )
    print(f"  Paired human:  {model_db.height:,} rows")

    model_db = standardise_gene_names(model_db)
    model_db = deduplicate(model_db)
    print(f"  After donglab sort: {model_db.height:,} rows")

    model_db = replace_na_strings(model_db)
    print(f"  After NA filter: {model_db.height:,} rows")

    print("Validating with pyrepseq (functional TCR check)...")
    model_db = validate_with_pyrepseq(model_db)
    print(f"  After validation: {model_db.height:,} rows")

    before_b2m = model_db.filter(
        pl.col("mhc_a").is_not_null() & pl.col("mhc_b").is_null()
    ).height
    model_db = fill_class_i_b2m(model_db)
    after_b2m = model_db.filter(
        pl.col("mhc_a").is_not_null() & pl.col("mhc_b").is_null()
    ).height
    print(f"  stage 1: B2M-filled {before_b2m - after_b2m:,} class-I rows")

    if args.subsample_n is not None and args.subsample_n < model_db.height:
        print(f"Subsampling to {args.subsample_n:,} rows (seed={args.subsample_seed})")
        model_db = model_db.sample(n=args.subsample_n, seed=args.subsample_seed)

    print(f"\nDatabase breakdown:")
    print(model_db["database"].value_counts().sort("count", descending=True))

    output_path = args.output_dir / "01_merged_paired_tcrs.tsv"
    model_db.write_csv(output_path, separator="\t")
    print(f"\nWrote {model_db.height:,} rows to {output_path}")


if __name__ == "__main__":
    main()

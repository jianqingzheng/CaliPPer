"""
Step 3: Assign CDR1/2 features, MHC contacts, and individual weights.

Replaces the original ANARCI + EBI API + IMGT Domain-Gap-Align workflow
with pre-computed lookup tables:
  - CDR1/CDR2 from V-gene database (tcr_db.tsv)
  - MHC contact regions from IMGT-aligned lookup (mhc_imgt_contacts.tsv)
  - MHC allele name fixes from translation table (mhc_translation.csv)

Input:
  - <input_dir>/02_tcrs_with_folds.tsv

Resources (relative to project root):
  - resources/external/tcr_db.tsv
  - resources/internal/mhc_imgt_contacts.tsv
  - resources/internal/mhc_translation.csv

Output:
  - <output_dir>/03_tcrs_with_features.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl


# Hardcoded allele patches from notebooks/5_sf_assign_cdrs_and_mhcs.ipynb cell 95.
# Applied AFTER the translation-table join, BEFORE the contacts join. The
# translation table fixes typos; these patches are explicit allele substitutions
# for cases where the contacts table only has one nearby allele variant.
#
# Additional patches added to undo silent tidytcells 2.1.5 mis-mappings that
# weren't present when the notebook ran. These are downstream-of-formatter so
# REPLACEMENT_MHCS sees the *post-tidytcells* value:
#   HLA-A*24:01 → HLA-A*02:04:01 (tidytcells bug; would have stayed *24:01 in nb)
# To recover the original meaning, map the post-tidytcells form back to the
# canonical allele the notebook expected.
REPLACEMENT_MHCS: dict[str, str] = {
    # From notebook 5 cell 95 — applied to the original (pre-tidytcells) value
    "HLA-A*24:01": "HLA-A*24:02",
    "HLA-DPA1*01:01": "HLA-DPA1*01:03",
    "HLA-B*12:01": "HLA-B*44:02",
    "HLA-B*51:193": "HLA-B*51:01",
    # tidytcells 2.1.5 silent mis-mapping unwind
    "HLA-A*02:04:01": "HLA-A*24:02",
}


def fix_mhc_alleles(
    df: pl.DataFrame, translation_path: Path
) -> pl.DataFrame:
    """Apply MHC allele name corrections from translation table, then patch.

    Deduplicates the translation table on (mhc_a, mhc_b) and drops rows
    where both keys are null — these are no-op translations that, combined
    with nulls_equal=True on the join, cause every null-MHC row in df to
    fan out once per matching translation entry.

    After the table-driven fix, applies four hardcoded REPLACEMENT_MHCS
    patches per notebook 5. Without these, ~117 HLA-A*24:01 rows alone
    fall through the contacts join and get dropped.
    """
    translation = (
        pl.read_csv(translation_path)
        .select("mhc_a", "mhc_b", "fix_a", "fix_b")
        .filter(pl.col("mhc_a").is_not_null() | pl.col("mhc_b").is_not_null())
        .sort("mhc_a", "mhc_b", "fix_a", "fix_b", nulls_last=True)
        .unique(subset=["mhc_a", "mhc_b"], keep="first", maintain_order=True)
    )
    return (
        df.join(translation, on=["mhc_a", "mhc_b"], how="left", nulls_equal=True)
        .with_columns(
            pl.when(pl.col("fix_a").is_not_null())
            .then(pl.col("fix_a"))
            .otherwise(pl.col("mhc_a"))
            .alias("mhc_a"),
            pl.when(pl.col("fix_b").is_not_null())
            .then(pl.col("fix_b"))
            .otherwise(pl.col("mhc_b"))
            .alias("mhc_b"),
        )
        .drop("fix_a", "fix_b")
        .with_columns(
            pl.col("mhc_a").replace(REPLACEMENT_MHCS).alias("mhc_a"),
            pl.col("mhc_b").replace(REPLACEMENT_MHCS).alias("mhc_b"),
        )
    )


def _load_tcr_db(tcr_db_path: Path) -> pl.DataFrame:
    """Load and preprocess TCR database for CDR lookup."""
    db = pl.read_csv(tcr_db_path, separator="\t")
    db = db.filter((pl.col("organism") == "human") & (pl.col("region") == "V"))
    db = db.with_columns(
        pl.col("id").str.replace(r"\*\d+$", "").alias("gene_family"),
        pl.col("cdrs").str.split(";").list.get(0).str.replace_all(r"\.", "").alias("cdr1"),
        pl.col("cdrs").str.split(";").list.get(1).str.replace_all(r"\.", "").alias("cdr2"),
    )
    return db.select("gene_family", "chain", "cdr1", "cdr2")


def assign_cdrs_from_db(
    df: pl.DataFrame, tcr_db_path: Path
) -> pl.DataFrame:
    """Look up CDR1/CDR2 sequences from V-gene database."""
    tcr_db = _load_tcr_db(tcr_db_path)

    # Normalize V-gene names to gene family (strip allele) for matching
    df = df.with_columns(
        pl.col("v_a_gene").str.replace(r"\*\d+$", "").alias("v_a_norm"),
        pl.col("v_b_gene").str.replace(r"\*\d+$", "").alias("v_b_norm"),
    )

    # Alpha chain CDRs
    tcr_db_a = tcr_db.filter(pl.col("chain") == "A").group_by("gene_family").first()
    df = df.join(
        tcr_db_a.select(
            pl.col("gene_family").alias("v_a_norm"),
            pl.col("cdr1").alias("cdr1_a"),
            pl.col("cdr2").alias("cdr2_a"),
        ),
        on="v_a_norm",
        how="left",
    )

    # Beta chain CDRs
    tcr_db_b = tcr_db.filter(pl.col("chain") == "B").group_by("gene_family").first()
    df = df.join(
        tcr_db_b.select(
            pl.col("gene_family").alias("v_b_norm"),
            pl.col("cdr1").alias("cdr1_b"),
            pl.col("cdr2").alias("cdr2_b"),
        ),
        on="v_b_norm",
        how="left",
    )

    df = df.drop("v_a_norm", "v_b_norm")

    # CDR3 is already in the data as cdr3_a_aa / cdr3_b_aa — rename for consistency
    if "cdr3_a_aa" in df.columns:
        df = df.rename({"cdr3_a_aa": "cdr3_a"})
    if "cdr3_b_aa" in df.columns:
        df = df.rename({"cdr3_b_aa": "cdr3_b"})

    return df


def join_mhc_contacts(
    df: pl.DataFrame, contacts_path: Path
) -> pl.DataFrame:
    """Join pre-computed MHC IMGT contact features.

    Tries exact match on (mhc_a, mhc_b), then falls back to 2-field
    allele resolution (e.g. HLA-A*02:01:48 -> HLA-A*02:01).
    """
    contacts = pl.read_csv(contacts_path, separator="\t")

    joined = df.join(contacts, on=["mhc_a", "mhc_b"], how="left")

    unmatched = joined.filter(pl.col("contact1").is_null())
    if unmatched.height > 0:
        # B2M is the β2-microglobulin gene symbol, not an HLA allele; the
        # 2-field regex would collapse it to None, breaking the join key on
        # both sides. Preserve it verbatim so class-I rows can match across
        # 3-field-suffix variants in the contacts table.
        truncate_a = (
            pl.when(pl.col("mhc_a") == "B2M")
            .then(pl.col("mhc_a"))
            .otherwise(pl.col("mhc_a").str.extract(r"^((?:HLA-)?[A-Z0-9]+\*\d+:\d+)"))
            .alias("mhc_a_2f")
        )
        truncate_b = (
            pl.when(pl.col("mhc_b") == "B2M")
            .then(pl.col("mhc_b"))
            .otherwise(pl.col("mhc_b").str.extract(r"^((?:HLA-)?[A-Z0-9]+\*\d+:\d+)"))
            .alias("mhc_b_2f")
        )

        # Deterministic contacts_2f construction: full-column sort then
        # unique(keep='first', maintain_order=True) so the picked
        # representative is stable across thread counts, Polars versions,
        # and innocuous row-order changes in mhc_imgt_contacts.tsv. Matches
        # the convention used by stage-1 deduplicate.
        contacts_with_2f = contacts.with_columns(truncate_a, truncate_b)
        contacts_2f = contacts_with_2f.sort(
            *sorted(contacts_with_2f.columns), nulls_last=True
        ).unique(
            subset=["mhc_a_2f", "mhc_b_2f"], keep="first", maintain_order=True
        )

        unmatched_rejoined = (
            unmatched.drop("contact1", "contact2", "pseudo_sequence")
            .with_columns(truncate_a.alias("mhc_a_2f"), truncate_b.alias("mhc_b_2f"))
            .join(
                contacts_2f.select("mhc_a_2f", "mhc_b_2f", "contact1", "contact2", "pseudo_sequence"),
                on=["mhc_a_2f", "mhc_b_2f"],
                how="left",
            )
            .drop("mhc_a_2f", "mhc_b_2f")
        )

        matched = joined.filter(pl.col("contact1").is_not_null())
        joined = pl.concat([matched, unmatched_rejoined], how="diagonal_relaxed")

    return joined


def compute_individual_weights(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-TCR individual weights, matching notebook 2 weighted_db cell.

    Group by (epitope, mhc_a, mhc_b) — NOT (epitope, cluster, mhc_a, mhc_b).
    Weight formula transcribed verbatim from the notebook:
        class_weight = len                          if len < 10
        class_weight = 10 * ln(len / ln(10))        otherwise
    Then individual_weight = class_weight / len.

    Note: the notebook helper `class_weight(n) = 10*np.log(n)/np.log(10)`
    is `10*log10(n)`, but the in-Polars expression has different parens
    yielding `10*ln(len/ln(10))`. We preserve the notebook's actual
    behavior so per-row weights match what JQ received.
    """
    log_log10 = float(np.log(np.log(10)))  # constant offset; pre-computed for Polars
    weights = (
        df.unique(
            subset=[
                "epitope", "cdr3_a", "v_a_gene", "j_a_gene",
                "cdr3_b", "v_b_gene", "j_b_gene",
                "database", "mhc_a", "mhc_b",
            ]
        )
        .group_by("epitope", "mhc_a", "mhc_b")
        .len()
        .with_columns(
            pl.when(pl.col("len") < 10)
            .then(pl.col("len").cast(pl.Float64))
            .otherwise(
                10.0 * (pl.col("len").cast(pl.Float64).log() - log_log10)
            )
            .alias("class_weight"),
        )
        .with_columns(
            (pl.col("class_weight") / pl.col("len")).alias("individual_weight")
        )
        .select("epitope", "mhc_a", "mhc_b", "individual_weight")
    )

    return df.join(weights, on=["epitope", "mhc_a", "mhc_b"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Assign CDR/MHC features and weights")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.project_root
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input_dir / "02_tcrs_with_folds.tsv"
    df = pl.read_csv(input_path, separator="\t")
    print(f"Loaded {df.height:,} rows from {input_path}")

    print("Fixing MHC allele names...")
    df = fix_mhc_alleles(df, root / "resources/internal/mhc_translation.csv")

    print("Assigning CDR1/CDR2 from V-gene database...")
    df = assign_cdrs_from_db(df, root / "resources/external/tcr_db.tsv")

    cdr_coverage = {
        "cdr1_a": df.filter(pl.col("cdr1_a").is_not_null()).height,
        "cdr1_b": df.filter(pl.col("cdr1_b").is_not_null()).height,
    }
    for k, v in cdr_coverage.items():
        print(f"  {k}: {v}/{df.height} ({v / df.height:.1%})")

    print("Joining MHC IMGT contact features...")
    df = join_mhc_contacts(df, root / "resources/internal/mhc_imgt_contacts.tsv")

    has_contacts = df.filter(pl.col("contact1").is_not_null()).height
    print(f"  MHC contacts: {has_contacts}/{df.height} ({has_contacts / df.height:.1%})")

    print("Computing individual weights...")
    df = compute_individual_weights(df)

    output_path = args.output_dir / "03_tcrs_with_features.tsv"
    df.write_csv(output_path, separator="\t")
    print(f"\nWrote {df.height:,} rows to {output_path}")


if __name__ == "__main__":
    main()

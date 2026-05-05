# TCR Data Preprocessing Pipeline

Reproduces the curated TCR–epitope binding dataset used in the S2DD/CaliPPer manuscript from raw public databases and internal data.

## Pipeline Overview

| Stage | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| 0 | `00_denoise_10x.py` | Raw 10x dextramer CSVs | `single_10x_binders.csv` | Denoise 10x UMI data via PCA + Leiden clustering + purity filtering |
| 1 | `01_merge_databases.py` | VDJdb, McPAS, 10x, DongLab | `01_merged_paired_tcrs.tsv` | Merge databases, filter to paired human αβ TCRs, validate with pyrepseq |
| 2 | `02_cluster_and_fold.py` | Merged TSV | `02_tcrs_with_folds.tsv` | Edit-distance clustering + hierarchical 5-fold assignment |
| 3 | `03_assign_features.py` | Clustered TSV | `03_tcrs_with_features.tsv` | Assign CDR1/2 from V-gene DB + MHC contact pseudo-sequences |
| 4 | `04_export_jq_format.py` | Annotated TSV | `train/val/test_data.csv` | Export: train (positives only), val/test (+ 5:1 shuffled negatives) |

## Quick Start

```bash
# Full pipeline (without 10x denoising)
./run_all.sh --project-root /path/to/tcr_ml

# With 10x denoising
./run_all.sh --project-root /path/to/tcr_ml --include-10x

# Subsampled run (for testing)
./run_all.sh --config subsample_n=1000 subsample_seed=42
```

## Stage Details

### Stage 0: 10x Denoising (optional)

Processes raw 10x Genomics CD8+ T-cell dextramer data from 4 healthy donors:
1. Normalise dextramer UMI counts (subtract max negative control, log-normalise)
2. PCA (8 components) on normalised dextramer space
3. kNN graph + Leiden clustering (hyperopt on k and resolution, maximising NMI)
4. Purity filter: keep clusters with >92% binding to a single epitope
5. Aggregate cells to clonotype level, filter to single-epitope binders

### Stage 1: Merge Databases

- **VDJdb** and **McPAS-TCR**: loaded from pre-formatted TSV files
- **10x Genomics**: UMI-denoised single-epitope binders (from Stage 0 or pre-formatted)
- **DongLab**: internal paired SARS-CoV-2 TCRs (6 epitopes: NP16, S34, S151, S174, M24, ORF3a)
- Filter to paired human αβ TCRs (both CDR3α and CDR3β non-null)
- Standardise gene nomenclature with pyrepseq (`tcr_enforce_functional=True`, `tcr_precision='allele'`)
- Restore B2M for class-I MHC rows (nulled by tidytcells validator)

### Stage 2: Cluster and Fold Assignment

- Group similar TCRs via connected components: combined α+β CDR3 edit distance <4, identical Vα and Vβ genes, and same epitope specificity
- Hierarchical fold assignment: 10x epitopes restricted to folds 0–2; other epitopes spread across 4 of 5 folds
- Rebalancing: single row indices moved between folds to equalise fold sizes (may break cluster integrity at boundaries)

### Stage 3: Feature Assignment

- CDR1/CDR2 sequences assigned from V-gene lookup table (`tcr_db.tsv`)
- MHC contact regions and pseudo-sequences from IMGT-aligned lookup (`mhc_imgt_contacts.tsv`)
- MHC allele name corrections from translation table + hardcoded patches
- Rows missing any required feature (CDR1/2, contacts, pseudo-sequence) are dropped

### Stage 4: Export

- CDR3 sequences trimmed: strip conserved N-terminal cysteine (C) and C-terminal phenylalanine/tryptophan (F/W) per IMGT convention
- **Train** (folds 0–2): positives only (binder=1.0)
- **Validation** (fold 3): positives + 5:1 epitope-shuffled negatives
- **Test** (fold 4): positives + 5:1 epitope-shuffled negatives
- Negatives: cross-join CDR features with mismatched epitope+MHC contacts, sample at target ratio

## Output Format

Each CSV has columns: `epitope, cdr1_a, cdr2_a, cdr3_a, cdr1_b, cdr2_b, cdr3_b, contact1, contact2, pseudo_sequence, binder`

## Dependencies

See `pyproject.toml`. Key: `polars>=1.0`, `pyrepseq>=1.5`, `python-igraph>=0.11`, `leidenalg>=0.10`, `tidytcells>=2.1,<2.2`.

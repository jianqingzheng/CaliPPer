# BigMHC Retrospective Study — Data Inventory

**Date:** 2026-04-20
**Source:** Albert et al., Nature Machine Intelligence 2023 (DOI: 10.1038/s42256-023-00694-6)
**Repo:** https://github.com/KarchinLab/bigmhc (cloned to `Model/BigMHC/`)
**Data:** Mendeley DOI: 10.17632/dvmz6pkzvb (v4)

---

## Data Files

| File | Rows | Description |
|------|------|-------------|
| `manafest.csv` | 837 | MANAFEST clinical validation (167 immunogenic, 670 non-immunogenic) |
| `im_train.csv` | 6,185 | Immunogenicity training data (1,407 pos, 4,778 neg) |
| `im_test.csv` | 937 | Immunogenicity test with pre-computed predictions from 10+ models |
| `im_val.csv` | — | Immunogenicity validation |
| `el_train.csv` | — | Eluted ligand training (45K+ positives) |
| `el_test.csv` | — | Eluted ligand test |
| `iedb.csv` | — | IEDB immunogenicity data |
| `pseudoseqs.csv` | 18,929 | HLA pseudosequence one-hot encoding |

## MANAFEST Clinical Data

- **Columns:** mhc, pep, tgt, wtp (wild-type peptide), gene
- **HLA alleles:** 43 unique
- **Peptides:** 830 unique mutant peptides
- **Labels:** tgt=1 (immunogenic by MANAFEST T-cell expansion assay), tgt=0 (non-immunogenic)
- **Source:** Johns Hopkins NSCLC patients, Smith/Anagnostou lab

## Training Overlap

- MANAFEST ∩ IM Train peptides: **0** (after merge on mhc+pep key)
- MANAFEST ∩ IM Test: **834/837** rows (MANAFEST ⊂ IM Test)
- All predictions pre-computed — no BigMHC inference needed

## Pre-computed Model Predictions (in im_test.csv)

BigMHC_IM, BigMHC_EL, BigMHC_ELIM, NetMHCpan-4.1, MHCflurry-2.0, MHCnuggets-2.4.0, MixMHCpred-2.1/2.2, PRIME-1.0/2.0, TransPHLA, HLAthena + BigMHC ablation variants

## S2DD Settings

| Parameter | Value |
|-----------|-------|
| Chain | peptide only (1-chain) |
| Distance | Levenshtein |
| k, b, K | 0.1, 0.1, 50 |
| Training ref | im_train.csv (6,185 rows) |

**Note:** 2-chain (peptide+MHC) tested but sigma_C assigns 99.9% to MHC allele name, and Levenshtein on HLA nomenclature is not biologically meaningful. Peptide-only is more interpretable. Future: use MHC pseudosequence amino acid distance.

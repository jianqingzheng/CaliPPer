# BigMHC Retrospective Study — Data Inventory

**Date:** 2026-04-20
**Source:** Albert et al., Nature Machine Intelligence 2023 (DOI: 10.1038/s42256-023-00694-6)
**Repo:** https://github.com/KarchinLab/bigmhc (cloned to `Model/BigMHC/`)
**Data:** Mendeley DOI: 10.17632/dvmz6pkzvb (v4)

---

## Panel C/D vs Panel E test sets (dual-split)

BigMHC uses **different cal/test pairs** for performance prediction (Fig 6 Panel C/D) and recalibration (Fig 6 Panel E+):

- **Panel C/D (prediction):** **HLA halfsplit within MANAFEST** (~400 cal / ~434 test, same-domain split, seed=42)
- **Panel E+ (recalibration):** **im\_val** (688, author validation set, 0% peptide overlap) → **full MANAFEST** (834, independent cross-dataset)

Reason: MANAFEST is clinically tested neoantigens prioritised by computational scoring at the experimental-design stage, so it is selection-biased and unsuitable as a prediction target with im\_val as cal (the concept drift between benchmark peptides and clinical neoantigens violates the covariate-shift-only assumption; all 3 prediction methods give ~0.285 AUROC error). For prediction, HLA halfsplit within MANAFEST keeps cal and test in the same domain (AUROC error ~0.015). For recalibration, im\_val → MANAFEST is the true retrospective deployment scenario and Δ-metrics are well-defined regardless of selection bias. Full discussion: `docs/methodology/fig6_dual_split.md`.

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

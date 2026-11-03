#!/usr/bin/env python3
"""deepAntigen (Zhou et al.) — fetch & stage raw data.

Source: Zhou et al., deepAntigen — pan-specific TCR-neoantigen binding predictor.
Authors' code/data: https://github.com/JiangBioLab/deepAntigen
Clinical validation: Lowery et al., Science 2022 (100-neoantigen ELISPOT panel,
                     DOI 10.1126/science.abl5447)
Cross-cohort: ImmuneCODE-MIRA portal (Adaptive Biotechnologies, registration
              required) — SARS-CoV-2 epitope-binding TCRs.

Files staged into INPUT_DIR/:

  Direct downloads from deepAntigen GitHub:
    Data/tcr_seq/proc_files/deepantigen_data/train.csv
    Data/tcr_seq/proc_files/deepantigen_data/zero_shot_test.csv
    Data/tcr_seq/proc_files/deepantigen_data/majority/majority_training_dataset.csv

  Manual download (Adaptive Biotechnologies registration):
    Data/tcr_seq/proc_files/deepantigen_data/immunecode_sars.csv
       (~1.1M rows, ImmuneCODE-MIRA portal; user registration required)
       Alternative: skip — only needed for Panel C/D cross-cohort prediction.

  Manual download (Science supplementary):
    Data/retrospective_deepantigen/lowery2022_neoantigen_elispot.csv
       Lowery 2022 supplementary Table S3 (100 confirmed neoantigens + ELISPOT).

The script downloads what's accessible and documents the rest. Stage 0c-0f
of reproduce_fig6.sh will then run from these staged files.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import download, manual_step, get_input_dir  # noqa: E402


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/JiangBioLab/deepAntigen/main/data"

GITHUB_FILES = [
    # (github subpath, target subpath, description)
    ("Dataset_7/train.csv",
     "Data/tcr_seq/proc_files/deepantigen_data/train.csv",
     "deepAntigen Datasets 1-7 training (~28k pairs)"),
    ("Dataset_8/zero_shot_test.csv",
     "Data/tcr_seq/proc_files/deepantigen_data/zero_shot_test.csv",
     "deepAntigen Dataset 8 zero-shot test (1,714 pairs)"),
    ("majority/majority_training_dataset.csv",
     "Data/tcr_seq/proc_files/deepantigen_data/majority/majority_training_dataset.csv",
     "deepAntigen majority-shot training (23k pairs, shared with PanPep)"),
]

# Adaptive Biotechnologies ImmuneCODE — requires account; document only.
IMMUNECODE_URL = "https://clients.adaptivebiotech.com/pub/covid-2020"

# Lowery 2022 — supplementary file URLs change; document instead.
LOWERY_2022_URL = "https://www.science.org/doi/10.1126/science.abl5447"


def prepare() -> bool:
    INPUT_DIR = get_input_dir()
    print(f"\n[deepantigen] Preparing data under {INPUT_DIR}/")
    ok = True

    # 1. GitHub direct downloads — best-effort (deepAntigen repo URLs may change)
    for gh_subpath, target_subpath, desc in GITHUB_FILES:
        url = f"{GITHUB_RAW_BASE}/{gh_subpath}"
        target = INPUT_DIR / target_subpath
        if not download(url, target, desc=desc):
            # GitHub fallback didn't work — surface manual download instructions
            ok &= manual_step(
                desc, "https://github.com/JiangBioLab/deepAntigen", target,
                f"Clone the deepAntigen repo and copy data/{gh_subpath} to the target path.",
            )

    # 2. ImmuneCODE (registration required) — strictly optional for Panel C/D
    immunecode_target = (INPUT_DIR / "Data" / "tcr_seq" / "proc_files"
                                   / "deepantigen_data" / "immunecode_sars.csv")
    immunecode_present = manual_step(
        "ImmuneCODE-MIRA SARS-CoV-2 cohort",
        IMMUNECODE_URL, immunecode_target,
        "Adaptive Biotechnologies portal requires registration. Download "
        "ImmuneCODE-Release-002 (MIRA), filter to SARS-CoV-2 epitopes only, "
        "and save the resulting ~1.1M-row CSV at the target path. "
        "Only needed for Fig 6 Panel C/D cross-cohort prediction; "
        "Panel E ΔAUROC/ΔAP do not depend on this file.",
    )
    if not immunecode_present:
        print("    (immunecode_sars.csv is optional — Panel E will still reproduce)")

    # 3. Lowery 2022 ELISPOT neoantigen panel
    lowery_target = (INPUT_DIR / "Data" / "retrospective_deepantigen"
                               / "lowery2022_neoantigen_elispot.csv")
    lowery_present = manual_step(
        "Lowery 2022 ELISPOT 100-neoantigen panel",
        LOWERY_2022_URL, lowery_target,
        "Download Lowery et al. Science 2022 supplementary Table S3, "
        "convert to CSV with columns ID, peptide, confirmed, save to target path. "
        "Required for Stage 0c neoantigen recalibration.",
    )
    if not lowery_present:
        ok = False

    if ok:
        print("[deepantigen] ✓ all required files staged")
    else:
        print("[deepantigen] ⚠ some files require manual download (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if prepare() else 1)

#!/usr/bin/env python3
"""BigMHC (Albert et al. Nature MI 2023) — fetch & stage raw data.

Source: Albert et al., Nature Machine Intelligence 2023, DOI 10.1038/s42256-023-00694-6
Authors' code: https://github.com/KarchinLab/bigmhc
Authors' data: Mendeley Data DOI 10.17632/dvmz6pkzvb (version 4)

Files staged into INPUT_DIR/:

  Direct downloads from GitHub:
    Model/BigMHC/data/pseudoseqs.csv          (~600 KB)

  Mendeley dataset (single ZIP, requires direct file URL):
    Data/retrospective_bigmhc/mendeley_data/BigMHC_Training_and_Evaluation_Data.zip
       → unzipped into Data/retrospective_bigmhc/mendeley_data/extracted/:
         manafest.csv, im_train.csv, im_val.csv, im_test.csv, el_train.csv,
         el_test.csv, iedb.csv

Mendeley files have direct URLs from their CDN once you have the file ID;
those IDs are recorded below. If the URL form changes (Mendeley sometimes
updates its file IDs), the script prints the manual download URL.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import download, manual_step, unzip_to, get_input_dir  # noqa: E402

GITHUB_PSEUDOSEQS = (
    "https://raw.githubusercontent.com/KarchinLab/bigmhc/master/data/pseudoseqs.csv"
)

# Mendeley dataset v4 — DOI resolves to data.mendeley.com/datasets/dvmz6pkzvb/4
# The full bundle ZIP file URL on Mendeley's CDN. If browsers prompt for download,
# the manual step block below will surface that.
MENDELEY_ZIP_URL = (
    "https://data.mendeley.com/api/datasets/dvmz6pkzvb/files-by-version/4/download"
)
MENDELEY_PAGE_URL = "https://data.mendeley.com/datasets/dvmz6pkzvb/4"


def prepare() -> bool:
    INPUT_DIR = get_input_dir()
    print(f"\n[bigmhc] Preparing data under {INPUT_DIR}/")
    ok = True

    # 1. GitHub direct download
    ok &= download(
        GITHUB_PSEUDOSEQS,
        INPUT_DIR / "Model" / "BigMHC" / "data" / "pseudoseqs.csv",
        desc="BigMHC pseudoseqs.csv (HLA pseudosequence one-hot encoding)",
    )

    # 2. Mendeley ZIP — try direct API URL first; if it fails, fall back to manual
    zip_target = (INPUT_DIR / "Data" / "retrospective_bigmhc" / "mendeley_data"
                            / "BigMHC_Training_and_Evaluation_Data.zip")
    extracted_dir = zip_target.parent / "extracted"

    expected_csvs = ["manafest.csv", "im_train.csv", "im_val.csv", "im_test.csv"]
    have_extracted = all((extracted_dir / f).exists() for f in expected_csvs)

    if have_extracted:
        print("  ✓ Mendeley extracted CSVs already present — skipping ZIP fetch + unzip")
    else:
        downloaded = download(
            MENDELEY_ZIP_URL, zip_target,
            desc="BigMHC Mendeley v4 bundle (~120 MB, manafest+im+el+iedb CSVs)",
        )
        if not downloaded:
            ok &= manual_step(
                "BigMHC Mendeley v4 bundle",
                MENDELEY_PAGE_URL, zip_target,
                "Visit the Mendeley page, click 'Download All Files' (~120 MB), "
                "rename to BigMHC_Training_and_Evaluation_Data.zip, and place at "
                "the target path.",
            )
        if zip_target.exists():
            ok &= unzip_to(zip_target, extracted_dir)
            # Verify expected files now exist
            for f in expected_csvs:
                fp = extracted_dir / f
                if not fp.exists():
                    print(f"  ✗ post-extract: {f} missing in {extracted_dir}/", file=sys.stderr)
                    ok = False
                else:
                    print(f"  ✓ extracted: {f}")

    if ok:
        print("[bigmhc] ✓ all files staged")
    else:
        print("[bigmhc] ✗ some files missing (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if prepare() else 1)

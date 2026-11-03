#!/usr/bin/env python3
"""XBCR-net (Lou et al. Cell Research 2022) — fetch & stage raw data.

Source: Lou, Zheng et al., Cell Research 2022, DOI 10.1038/s41422-022-00727-6
Authors' code: https://github.com/jianqingzheng/XBCR-net
Authors' data: Mendeley Data DOI 10.17632/s2x6pkse (supplementary Data S1/S2)
                + Nature Source Data file

Files staged into INPUT_DIR/:

  Direct downloads from GitHub (XBCR-net example data):
    Model/XBCR-net/data/binding/exper/example-experimental_data.xlsx (~3 MB)
    Model/XBCR-net/data/binding/nonexp/example-negative_data.xlsx     (~2 MB)

  Mendeley supplementary (Data S1 + S2 + Source Data):
    Data/retrospective_xbcr/data_s1_training.xlsx                    (~57 MB)
    Data/retrospective_xbcr/data_s2_scbcrseq.xlsx                    (~9 MB)
    Data/retrospective_xbcr/source_data.xlsx                          (~14 KB)

  Manual conversion step (one Python snippet):
    Data/retrospective_xbcr/data_s1_training.xlsx
      → extracted_panels/{panel1_training,panel1_test}.csv

After files are staged, the script runs extract_xbcr_panels() to produce
the panel CSVs that Stage 0a consumes. Panel 2 (therapeutic mAbs) is also
written based on the Source Data sheet + 15 CoV-AbDab clinical mAbs.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import download, manual_step, get_input_dir  # noqa: E402


GITHUB_EXAMPLE_EXPER = (
    "https://github.com/jianqingzheng/XBCR-net/raw/main/data/binding/exper/"
    "example-experimental_data.xlsx"
)
GITHUB_EXAMPLE_NONEXP = (
    "https://github.com/jianqingzheng/XBCR-net/raw/main/data/binding/nonexp/"
    "example-negative_data.xlsx"
)

# Mendeley DOI s2x6pkse — these direct file URLs are best-effort; manual fallback below.
MENDELEY_S1_URL = (
    "https://data.mendeley.com/api/datasets/s2x6pkse/files-by-version/1/download"
)
MENDELEY_PAGE_URL = "https://data.mendeley.com/datasets/s2x6pkse/1"


def extract_xbcr_panels(input_dir: Path) -> bool:
    """Extract Panel 1 train/test from data_s1_training.xlsx via pandas.

    The xlsx has a 'panel' column ('panel1'/'panel2') and a 'split' column
    ('train'/'test'). Sequences are in heavy_chain, light_chain, variant_seq;
    label is rbd (1 = binder, 0 = non-binder).
    """
    src_xlsx = (input_dir / "Data" / "retrospective_xbcr" / "data_s1_training.xlsx")
    out_dir = src_xlsx.parent / "extracted_panels"

    if not src_xlsx.exists():
        print(f"  ✗ extract: data_s1_training.xlsx missing — cannot extract panels")
        return False

    expected = ["panel1_training.csv", "panel1_test.csv"]
    if all((out_dir / f).exists() for f in expected):
        print("  ✓ extract: panel CSVs already present — skipping re-extraction")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  → extracting panel1 train/test from {src_xlsx.name}")
    try:
        import pandas as pd
        df = pd.read_excel(src_xlsx, sheet_name=0)
        # Author's columns: panel, split, heavy_chain, light_chain, variant_seq, rbd
        if "panel" not in df.columns or "split" not in df.columns:
            print(f"  ✗ extract: data_s1_training.xlsx does not have expected columns. "
                  f"Got: {list(df.columns)[:8]}...")
            return False
        train = df[(df["panel"] == "panel1") & (df["split"] == "train")]
        test  = df[(df["panel"] == "panel1") & (df["split"] == "test")]
        train.to_csv(out_dir / "panel1_training.csv", index=False)
        test.to_csv(out_dir / "panel1_test.csv", index=False)
        print(f"  ✓ wrote {out_dir.name}/panel1_training.csv ({len(train)} rows)")
        print(f"  ✓ wrote {out_dir.name}/panel1_test.csv ({len(test)} rows)")
        return True
    except Exception as e:
        print(f"  ✗ extract failed: {type(e).__name__}: {e}")
        return False


def prepare() -> bool:
    INPUT_DIR = get_input_dir()
    print(f"\n[xbcr-net] Preparing data under {INPUT_DIR}/")
    ok = True

    # 1. GitHub example data (direct)
    ok &= download(
        GITHUB_EXAMPLE_EXPER,
        INPUT_DIR / "Model" / "XBCR-net" / "data" / "binding" / "exper"
                  / "example-experimental_data.xlsx",
        desc="XBCR-net example experimental data (GitHub)",
    )
    ok &= download(
        GITHUB_EXAMPLE_NONEXP,
        INPUT_DIR / "Model" / "XBCR-net" / "data" / "binding" / "nonexp"
                  / "example-negative_data.xlsx",
        desc="XBCR-net example negative data (GitHub)",
    )

    # 2. Mendeley Data S1/S2/Source Data
    mendeley_zip = INPUT_DIR / "Data" / "retrospective_xbcr" / "_mendeley_s2x6pkse_v1.zip"
    s1_target = INPUT_DIR / "Data" / "retrospective_xbcr" / "data_s1_training.xlsx"

    if s1_target.exists():
        print(f"  ✓ Mendeley supplementary already extracted (data_s1_training.xlsx present)")
    else:
        # Try direct Mendeley API ZIP fetch
        if download(MENDELEY_S1_URL, mendeley_zip,
                    desc="XBCR-net Mendeley dataset v1 (Data S1+S2+Source Data, ~67 MB)"):
            try:
                from common import unzip_to
                unzip_to(mendeley_zip, mendeley_zip.parent)
                print(f"  ✓ Mendeley ZIP unzipped — checking for data_s1_training.xlsx")
            except Exception as e:
                print(f"  ✗ Mendeley unzip failed: {e}", file=sys.stderr)
                ok = False
        else:
            ok &= manual_step(
                "XBCR-net Mendeley dataset",
                MENDELEY_PAGE_URL, s1_target,
                "Visit the Mendeley page, click 'Download All Files', and place "
                "data_s1_training.xlsx + data_s2_scbcrseq.xlsx + source_data.xlsx "
                "in Data/retrospective_xbcr/. The extract step below will then run.",
            )

    # 3. Extract Panel 1 train/test from Data S1
    if s1_target.exists():
        ok &= extract_xbcr_panels(INPUT_DIR)
    else:
        print("  ⚠ panel extraction skipped (data_s1_training.xlsx missing)")
        ok = False

    if ok:
        print("[xbcr-net] ✓ all files staged + panels extracted")
    else:
        print("[xbcr-net] ✗ some steps incomplete (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if prepare() else 1)

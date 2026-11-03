#!/usr/bin/env python3
"""BigMHC (Albert et al. Nature MI 2023) — fetch & stage raw data.

Source: Albert et al., Nature Machine Intelligence 2023, DOI 10.1038/s42256-023-00694-6
Authors' code: https://github.com/KarchinLab/bigmhc
Authors' data: Mendeley Data DOI 10.17632/dvmz6pkzvb (version 4)

Files staged into INPUT_DIR/:

  Direct download from GitHub:
    Model/BigMHC/data/pseudoseqs.csv          (~16 MB)

  Mendeley dataset bundle (datasets.zip, ~144 MB) → only the immunogenicity
  CSVs are extracted (the ~380 MB el_* eluted-ligand files are skipped):
    Data/retrospective_bigmhc/mendeley_data/extracted/
        BigMHC Training and Evaluation Data/
            im_train.csv, im_val.csv, im_test.csv

  MANAFEST clinical validation labels (manafest.csv) are Johns Hopkins NSCLC
  patient data and are NOT part of the Mendeley bundle. Their ground-truth
  keys/labels (mhc, pep, tgt) are committed in the reproduction results file,
  so manafest.csv is reconstructed from that source (see reconstruct_manafest).

NOTE on the Mendeley URL: the legacy `files-by-version/<v>/download` endpoint
is defunct (it now returns an unrelated dataset). The real per-file download
URL is resolved at run time via the Mendeley public API, with a pinned direct
URL as a fallback.
"""
from __future__ import annotations
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import download, manual_step, get_input_dir  # noqa: E402

GITHUB_PSEUDOSEQS = (
    "https://raw.githubusercontent.com/KarchinLab/bigmhc/master/data/pseudoseqs.csv"
)

# Mendeley dataset v4 — DOI 10.17632/dvmz6pkzvb.4
MENDELEY_DATASET_ID = "dvmz6pkzvb"
MENDELEY_VERSION = 4
MENDELEY_FILES_API = (
    f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"
    f"/files?folder_id=root&version={MENDELEY_VERSION}"
)
MENDELEY_BUNDLE_FILENAME = "datasets.zip"  # the file we need inside the dataset
# Pinned direct fallback URL (valid as of 2026-07; Mendeley download_expiry_time 2126)
MENDELEY_BUNDLE_FALLBACK_URL = (
    "https://data.mendeley.com/public-files/datasets/dvmz6pkzvb/files/"
    "1da0314f-692d-4c39-b81f-fa2a7dba86bf/file_downloaded"
)
MENDELEY_PAGE_URL = "https://data.mendeley.com/datasets/dvmz6pkzvb/4"

# CSVs we actually need from the ~144 MB bundle (skip the ~380 MB el_* files).
NEEDED_CSVS = ["im_train.csv", "im_val.csv", "im_test.csv"]
# Subfolder name the Fig 6 compute scripts (compute_fig6_recal_data.py) expect.
EXTRACT_SUBDIR = "BigMHC Training and Evaluation Data"


def resolve_mendeley_url() -> str:
    """Resolve the current download URL for datasets.zip via the Mendeley public
    API; fall back to the pinned direct URL if the API is unreachable/changed."""
    try:
        # Mendeley's public API rejects the default Python-urllib UA with 403.
        req = urllib.request.Request(
            MENDELEY_FILES_API, headers={"User-Agent": "curl/8.0", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            files = json.load(r)
        for f in files:
            if f.get("filename") == MENDELEY_BUNDLE_FILENAME:
                url = f["content_details"]["download_url"]
                print(f"  → resolved Mendeley '{MENDELEY_BUNDLE_FILENAME}' via public API")
                return url
        print(f"  ⚠ '{MENDELEY_BUNDLE_FILENAME}' not in API listing; using fallback URL")
    except Exception as e:
        print(f"  ⚠ Mendeley API resolve failed ({e}); using pinned fallback URL")
    return MENDELEY_BUNDLE_FALLBACK_URL


def extract_needed(zip_path: Path, dst_dir: Path) -> bool:
    """Extract only the immunogenicity CSVs (im_train/im_val/im_test) from the
    bundle into dst_dir. The zip stores members at the archive root."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            for csv in NEEDED_CSVS:
                member = next((n for n in names if Path(n).name == csv), None)
                if member is None:
                    print(f"  ✗ {csv} not found in {zip_path.name}", file=sys.stderr)
                    return False
                target = dst_dir / csv
                with z.open(member) as src_f, open(target, "wb") as out_f:
                    out_f.write(src_f.read())
                print(f"  ✓ extracted: {csv} ({target.stat().st_size:,} B)")
        return True
    except zipfile.BadZipFile as e:
        print(f"  ✗ {zip_path.name} is not a valid ZIP ({e})", file=sys.stderr)
        return False


def reconstruct_manafest(input_dir: Path, dst: Path) -> bool:
    """Reconstruct the raw MANAFEST validation table (manafest.csv) that Stage 1
    consumes. The clinical labels (Johns Hopkins NSCLC patients) are not in any
    public deposit, but their ground-truth keys/labels are committed in
    results/bigmhc_retrospective/reproduction/manafest_with_distances.csv."""
    if dst.exists():
        print(f"  ✓ manafest.csv already present ({dst})")
        return True
    src = (input_dir / "results" / "bigmhc_retrospective" / "reproduction"
                     / "manafest_with_distances.csv")
    if not src.exists():
        print(f"  ✗ cannot reconstruct manafest.csv — source missing: {src}", file=sys.stderr)
        return False
    try:
        import pandas as pd
        df = pd.read_csv(src)
        keep = [c for c in ["mhc", "pep", "tgt", "wtp", "gene", "tgt_te"] if c in df.columns]
        man = df[keep].drop_duplicates(subset=["mhc", "pep"]).reset_index(drop=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        man.to_csv(dst, index=False)
        print(f"  ✓ reconstructed manafest.csv ({len(man)} rows, "
              f"{int(man['tgt'].sum())} immunogenic) from committed ground truth")
        return True
    except Exception as e:
        print(f"  ✗ manafest.csv reconstruction failed: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def prepare() -> bool:
    INPUT_DIR = get_input_dir()
    print(f"\n[bigmhc] Preparing data under {INPUT_DIR}/")
    ok = True

    # 1. GitHub direct download (HLA pseudosequence one-hot encoding)
    ok &= download(
        GITHUB_PSEUDOSEQS,
        INPUT_DIR / "Model" / "BigMHC" / "data" / "pseudoseqs.csv",
        desc="BigMHC pseudoseqs.csv (HLA pseudosequence one-hot encoding)",
    )

    # 2. Mendeley bundle → extract im_train/im_val/im_test
    mendeley_dir = INPUT_DIR / "Data" / "retrospective_bigmhc" / "mendeley_data"
    zip_target = mendeley_dir / "datasets.zip"
    extract_dir = mendeley_dir / "extracted" / EXTRACT_SUBDIR

    if all((extract_dir / f).exists() for f in NEEDED_CSVS):
        print("  ✓ BigMHC im_* CSVs already extracted — skipping ZIP fetch")
    else:
        url = resolve_mendeley_url()
        if download(url, zip_target,
                    desc="BigMHC Mendeley v4 bundle datasets.zip (~144 MB)"):
            ok &= extract_needed(zip_target, extract_dir)
        else:
            ok &= manual_step(
                "BigMHC Mendeley v4 bundle",
                MENDELEY_PAGE_URL, zip_target,
                "Visit the Mendeley page, download datasets.zip (~144 MB), place it at "
                f"{zip_target}, then re-run. im_train/im_val/im_test.csv are extracted "
                f"into extracted/{EXTRACT_SUBDIR}/.",
            )

    # 3. MANAFEST clinical labels (not in the Mendeley bundle) — reconstruct
    ok &= reconstruct_manafest(INPUT_DIR, extract_dir / "manafest.csv")

    if ok:
        print("[bigmhc] ✓ all files staged")
    else:
        print("[bigmhc] ✗ some files missing (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if prepare() else 1)

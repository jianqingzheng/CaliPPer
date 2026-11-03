#!/usr/bin/env python3
"""PanPep (Gao et al. Nature MI 2023) — fetch & stage raw data.

Source: Gao et al., Nature Machine Intelligence 2023, DOI 10.1038/s42256-023-00619-3
Authors' code+data: https://github.com/bm2-lab/PanPep
Authors' Zenodo: DOI 10.5281/zenodo.7544387 (PanPep-v1.0.0.zip = GitHub snapshot)

Zenodo record 7544387 contains:
  - PanPep-v1.0.0.zip       (~810 KB) — full repo snapshot incl. Data/ + Requirements/
  - Control dataset.txt     (~880 MB) — control TCR pool (not needed for Fig 6 Panel E)

The script downloads + unzips PanPep-v1.0.0.zip and stages:
  Model/PanPep/Data/{meta_dataset,base_dataset,zero_dataset}.csv
  Model/PanPep/Requirements/{model.pt, Content_memory.pkl, Query.pkl}

No manual step required.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import download, unzip_to, get_input_dir  # noqa: E402

ZENODO_PANPEP_ZIP = (
    "https://zenodo.org/records/7544387/files/bm2-lab/PanPep-v1.0.0.zip"
)

EXPECTED_FILES = [
    "Data/meta_dataset.csv",
    "Data/base_dataset.csv",
    "Data/zero_dataset.csv",
    "Requirements/model.pt",
    "Requirements/Content_memory.pkl",
    "Requirements/Query.pkl",
]


def prepare() -> bool:
    INPUT_DIR = get_input_dir()
    print(f"\n[panpep] Preparing data under {INPUT_DIR}/Model/PanPep/")

    panpep_dst = INPUT_DIR / "Model" / "PanPep"

    # Quick exit if all expected files already present
    if all((panpep_dst / f).exists() for f in EXPECTED_FILES):
        print(f"  ✓ all 6 expected files already present — skipping ZIP fetch")
        print(f"[panpep] ✓ all files staged")
        return True

    # Download + unzip
    zip_target = panpep_dst.parent / "_PanPep-v1.0.0.zip"
    if not download(
        ZENODO_PANPEP_ZIP, zip_target,
        desc="PanPep Zenodo PanPep-v1.0.0.zip (~810 KB, GitHub repo snapshot incl. Data + Requirements)",
    ):
        return False

    extract_root = panpep_dst.parent / "_panpep_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    if not unzip_to(zip_target, extract_root):
        return False

    # The ZIP contains a top-level directory like "bm2-lab-PanPep-XXXXXXX/"
    # Find it and copy its Data/ + Requirements/ to Model/PanPep/
    top_level_dirs = [p for p in extract_root.iterdir() if p.is_dir()]
    if not top_level_dirs:
        print(f"  ✗ unzip produced no directories in {extract_root}/")
        return False
    repo_root = top_level_dirs[0]
    print(f"  → extracted repo root: {repo_root.name}")

    ok = True
    panpep_dst.mkdir(parents=True, exist_ok=True)
    for subpath in EXPECTED_FILES:
        src = repo_root / subpath
        dst = panpep_dst / subpath
        if not src.exists():
            print(f"  ✗ missing in repo snapshot: {subpath}")
            ok = False
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  ✓ staged: {subpath} ({dst.stat().st_size:,} B)")

    # Cleanup
    shutil.rmtree(extract_root, ignore_errors=True)
    zip_target.unlink(missing_ok=True)

    if ok:
        print("[panpep] ✓ all 6 files staged")
    else:
        print("[panpep] ✗ some files missing in repo snapshot")
    return ok


if __name__ == "__main__":
    sys.exit(0 if prepare() else 1)

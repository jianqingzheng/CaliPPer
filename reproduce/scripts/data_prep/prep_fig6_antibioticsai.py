#!/usr/bin/env python3
"""AntibioticsAI (Wong et al. Nature 2024) — fetch & stage raw data.

Source: Wong et al., Nature 2024, DOI 10.1038/s41586-023-06887-8
Authors' code: https://github.com/felix-wong/antibiotics-deep-learning
                (Zenodo DOI 10.5281/zenodo.10095879)

Files staged into INPUT_DIR/Data/retrospective_antibioticsai/ and
INPUT_DIR/Model/AntibioticsAI/:
  - supplementary/41586_2023_6887_MOESM4_ESM.xlsx  (Nature direct URL, ~3 MB)
  - working_example/train.csv                        (GitHub raw, ~3 MB)

Both are direct downloads — no manual step required.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import download, unzip_to, get_input_dir  # noqa: E402

NATURE_XLSX = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06887-8/"
    "MediaObjects/41586_2023_6887_MOESM4_ESM.xlsx"
)
# Authors' code+data on Zenodo (DOI 10.5281/zenodo.10095879). working_example.zip
# is ~440 KB and contains train.csv (~39k labelled SMILES) + test.csv + hit.csv.
ZENODO_WORKING_EXAMPLE_ZIP = (
    "https://zenodo.org/records/10095879/files/working_example.zip"
)


def prepare() -> bool:
    INPUT_DIR = get_input_dir()
    print(f"\n[antibioticsai] Preparing data under {INPUT_DIR}/")
    ok = True

    ok &= download(
        NATURE_XLSX,
        INPUT_DIR / "Data" / "retrospective_antibioticsai" / "supplementary"
                  / "41586_2023_6887_MOESM4_ESM.xlsx",
        desc="Nature 2024 supplementary MOESM4 (AntibioticsAI test sets + predictions)",
    )

    # Zenodo working_example.zip → Model/AntibioticsAI/working_example/{train,test,hit}.csv
    zip_target = INPUT_DIR / "Model" / "AntibioticsAI" / "_working_example.zip"
    extract_dir = INPUT_DIR / "Model" / "AntibioticsAI"
    train_csv = extract_dir / "working_example" / "train.csv"

    if train_csv.exists():
        print(f"  ✓ AntibioticsAI working_example/train.csv: already extracted")
    else:
        if download(
            ZENODO_WORKING_EXAMPLE_ZIP, zip_target,
            desc="AntibioticsAI Zenodo working_example.zip (~440 KB, contains train/test/hit CSVs)",
        ):
            if unzip_to(zip_target, extract_dir):
                if train_csv.exists():
                    print(f"  ✓ extracted: working_example/train.csv ({train_csv.stat().st_size:,} B)")
                    # Clean up the zip after successful extraction
                    zip_target.unlink()
                else:
                    print(f"  ✗ extract: train.csv missing in {extract_dir}/working_example/")
                    ok = False
            else:
                ok = False
        else:
            ok = False

    if ok:
        print("[antibioticsai] ✓ all required files staged")
    else:
        print("[antibioticsai] ✗ some downloads failed (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if prepare() else 1)

"""Common helpers for Fig 6 data preparation scripts.

These scripts download author-published data from original sources (Nature
supplementary, Mendeley datasets, Zenodo, GitHub) and stage it into
INPUT_DIR so Stage 0 of reproduce_fig6.sh can run from scratch.

The alternative for reviewers who trust CaliPPer's pre-packaged deposit
is `reproduce/[retired]`, which fetches the same files already
extracted into a tarball from CaliPPer's own Zenodo record.
"""
from __future__ import annotations
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, expected_sha256: Optional[str] = None,
             desc: Optional[str] = None) -> bool:
    """Download `url` to `target` unless target already exists with matching sha256.

    Returns True on success, False on hard failure. Prints status to stdout.
    """
    label = desc or target.name
    if target.exists():
        if expected_sha256 is None:
            print(f"  ✓ {label}: already present ({target})")
            return True
        got = sha256_of(target)
        if got == expected_sha256:
            print(f"  ✓ {label}: already present + sha256 verified")
            return True
        print(f"  ⚠ {label}: present but sha256 mismatch — re-downloading")
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → downloading {label} from {url}")
    try:
        # Use curl with retries + redirect handling (more robust than urllib for big files)
        subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "3", "--retry-delay", "5",
            "-o", str(target), url,
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {label}: curl failed ({e})", file=sys.stderr)
        return False
    except FileNotFoundError:
        # curl not available — fallback to urllib
        try:
            urllib.request.urlretrieve(url, str(target))
        except Exception as e:
            print(f"  ✗ {label}: urlretrieve failed ({e})", file=sys.stderr)
            return False

    if expected_sha256:
        got = sha256_of(target)
        if got != expected_sha256:
            print(f"  ✗ {label}: sha256 mismatch (got {got[:12]}…, expected {expected_sha256[:12]}…)",
                  file=sys.stderr)
            return False
        print(f"  ✓ {label}: downloaded + sha256 verified ({target.stat().st_size:,} B)")
    else:
        print(f"  ✓ {label}: downloaded ({target.stat().st_size:,} B)")
    return True


def manual_step(label: str, url: str, target: Path, instructions: str) -> bool:
    """Document a manual download step. Returns True if target exists, False if missing."""
    if target.exists():
        print(f"  ✓ {label}: manual step already completed ({target.stat().st_size:,} B at {target})")
        return True
    print(f"  ✗ {label}: MANUAL DOWNLOAD REQUIRED")
    print(f"    Source URL: {url}")
    print(f"    Target path: {target}")
    print(f"    Instructions: {instructions}")
    return False


def unzip_to(zip_path: Path, dst_dir: Path) -> bool:
    """Unzip `zip_path` into `dst_dir` (creates if needed). Idempotent."""
    if not zip_path.exists():
        print(f"  ✗ unzip: source ZIP missing ({zip_path})", file=sys.stderr)
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"  → unzipping {zip_path.name} → {dst_dir}/")
    try:
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(dst_dir)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ unzip failed ({e})", file=sys.stderr)
        return False
    return True


def get_input_dir() -> Path:
    """Resolve INPUT_DIR via the standard CaliPPer _paths bootstrap."""
    here = Path(__file__).resolve()
    scripts_dir = here.parent.parent  # reproduce/scripts/
    sys.path.insert(0, str(scripts_dir))
    from _paths import INPUT_DIR  # noqa
    return Path(INPUT_DIR)

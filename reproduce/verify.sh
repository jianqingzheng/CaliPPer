#!/usr/bin/env bash
# verify.sh — compare regenerated figN_values.csv against committed reference
#
# The reproducibility GATE for CaliPPer. Compares every column of
# reproduce/data/output/figN_values.csv against reproduce/data/reference/figN_values.csv
# within 1e-10 floating-point tolerance.
#
# Figure rendering (PDFs/PNGs) is NOT verified — that varies with matplotlib/font/locale
# and is not portable across machines.
#
# Usage:
#   bash reproduce/verify.sh                       # all figures
#   bash reproduce/verify.sh --figure 2            # just fig2
#   bash reproduce/verify.sh --figure 2,4,6        # subset
#   bash reproduce/verify.sh --tolerance 1e-8      # custom tolerance (default 1e-10)
#
# Output:
#   stdout: per-figure PASS/FAIL summary
#   reproduce/verification.json: machine-readable report
#
# Exit codes: 0 = all PASS, 1 = at least one FAIL, 2 = bad args / missing files

set -uo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO_DIR")"

# Defaults
FIGURES="2,3,4,5,6"
TOLERANCE="1e-10"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --figure) FIGURES="$2"; shift 2 ;;
    --tolerance) TOLERANCE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done
[[ "$FIGURES" == "all" ]] && FIGURES="2,3,4,5,6"

# Delegate to Python (numerical comparison is easier in Python than bash)
python3 << PYEOF
import json
import sys
from pathlib import Path

ROOT = Path("$ROOT")
REPRO_DIR = Path("$REPRO_DIR")
TOLERANCE = float("$TOLERANCE")
FIGURES = "$FIGURES".split(",")

ref_dir = REPRO_DIR / "data" / "reference"
out_dir = REPRO_DIR / "data" / "output"
report = {"tolerance": TOLERANCE, "figures": {}, "overall_pass": True}

for fig_num in FIGURES:
    fig_num = fig_num.strip()
    ref_file = ref_dir / f"fig{fig_num}_values.csv"
    out_file = out_dir / f"fig{fig_num}_values.csv"

    entry = {"reference": str(ref_file), "output": str(out_file), "status": None}

    if not ref_file.exists():
        entry["status"] = "FAIL"
        entry["reason"] = f"Reference file missing: {ref_file}"
        report["figures"][f"fig{fig_num}"] = entry
        report["overall_pass"] = False
        print(f"  fig{fig_num}: ✗ FAIL — reference missing: {ref_file.name}")
        continue

    if not out_file.exists():
        entry["status"] = "FAIL"
        entry["reason"] = f"Output file missing — did reproduce.sh --figure {fig_num} run?"
        report["figures"][f"fig{fig_num}"] = entry
        report["overall_pass"] = False
        print(f"  fig{fig_num}: ✗ FAIL — output missing: {out_file.name} (run reproduce.sh first)")
        continue

    # Numerical comparison
    try:
        import pandas as pd
        import numpy as np
        ref_df = pd.read_csv(ref_file)
        out_df = pd.read_csv(out_file)

        if ref_df.shape != out_df.shape:
            entry["status"] = "FAIL"
            entry["reason"] = f"Shape mismatch: ref {ref_df.shape} vs out {out_df.shape}"
            report["overall_pass"] = False
            print(f"  fig{fig_num}: ✗ FAIL — shape mismatch ref {ref_df.shape} vs out {out_df.shape}")
            report["figures"][f"fig{fig_num}"] = entry
            continue

        if list(ref_df.columns) != list(out_df.columns):
            entry["status"] = "FAIL"
            entry["reason"] = f"Column mismatch: ref {list(ref_df.columns)} vs out {list(out_df.columns)}"
            report["overall_pass"] = False
            print(f"  fig{fig_num}: ✗ FAIL — column mismatch")
            report["figures"][f"fig{fig_num}"] = entry
            continue

        # Compare numerical columns within tolerance; string columns must match exactly
        max_abs_diff = 0.0
        max_diff_col = None
        for col in ref_df.columns:
            if pd.api.types.is_numeric_dtype(ref_df[col]):
                diff = np.abs(ref_df[col].values - out_df[col].values)
                # Handle NaN: NaN == NaN treated as equal
                mask = ~(np.isnan(ref_df[col].values) & np.isnan(out_df[col].values))
                if mask.any():
                    col_max = float(np.nanmax(diff[mask])) if mask.any() else 0.0
                    if col_max > max_abs_diff:
                        max_abs_diff = col_max
                        max_diff_col = col
            else:
                if not (ref_df[col].astype(str) == out_df[col].astype(str)).all():
                    entry["status"] = "FAIL"
                    entry["reason"] = f"String column '{col}' has mismatches"
                    report["overall_pass"] = False
                    print(f"  fig{fig_num}: ✗ FAIL — string mismatch in column '{col}'")
                    break

        if entry["status"] is None:
            entry["max_abs_diff"] = max_abs_diff
            entry["max_diff_col"] = max_diff_col
            if max_abs_diff <= TOLERANCE:
                entry["status"] = "PASS"
                print(f"  fig{fig_num}: ✓ PASS  (max |Δ| = {max_abs_diff:.2e})")
            else:
                entry["status"] = "FAIL"
                entry["reason"] = f"Max abs diff {max_abs_diff:.6e} > tolerance {TOLERANCE:.0e} (column '{max_diff_col}')"
                report["overall_pass"] = False
                print(f"  fig{fig_num}: ✗ FAIL — max |Δ| = {max_abs_diff:.2e} > {TOLERANCE:.0e} in column '{max_diff_col}'")
    except Exception as e:
        entry["status"] = "FAIL"
        entry["reason"] = f"Exception during comparison: {e}"
        report["overall_pass"] = False
        print(f"  fig{fig_num}: ✗ FAIL — exception: {e}")

    report["figures"][f"fig{fig_num}"] = entry

# Write machine-readable report
report_path = REPRO_DIR / "verification.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport written: {report_path}")

# Summary
print(f"\n{'='*40}")
total = len(report['figures'])
passed = sum(1 for e in report['figures'].values() if e['status'] == 'PASS')
print(f"{passed}/{total} figures PASS within tolerance {TOLERANCE:.0e}")
if report['overall_pass']:
    print("✓ ALL CLEAR — manuscript values reproduced")
    sys.exit(0)
else:
    print("✗ FAILURES — see verification.json for details")
    sys.exit(1)
PYEOF

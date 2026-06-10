#!/usr/bin/env python3
"""fig6.py - Figure 6 driver: Retrospective validation (5 published studies).

Orchestrates the **full end-to-end** Fig 6 reproduction in one command:

  Stage 1. ``compute_fig6_recal_data.py``  → per-study sample CSVs +
           ``recal_summary_all.csv`` at ``OUTPUT_DIR/recal_data/``
           (numerical values feeding Panel E + all per-study ROC/TDR
           panels). Output is BIT-EXACT (verified 9.99e-16 vs reference).
  Stage 2. ``compute_fig6_panel_c_d.py``   → Panel C prediction scatter
           data + Panel D 3-method comparison data.
  Stage 3. ``generate_fig6_redesign.py``   → 14 numerical panels (c–p)
           rendered to ``{out_fig}/fig6_{label}_{name}.{png,pdf}``.
           Panels a/b are placeholders here — see "Panels A/B" note below.
  Stage 4. Extract ``recal_summary_all.csv`` (long) to ``fig6_values.csv``
           (wide canonical schema for ``verify.sh``).

Canonical schema (one row per study):

    study, n, raw_aucroc, cal_aucroc, delta_aucroc,
           raw_ap, cal_ap, delta_ap

Rows are ordered by ``delta_aucroc`` descending, matching the manuscript
Fig 6 Panel E dumbbell ordering: XBCR-net, deepAntigen, AntibioticsAI,
BigMHC, PanPep.

Verification target: ``data/reference/fig6_values.csv``. ``verify.sh``
compares the regenerated values against the reference within 1e-10
floating-point tolerance.

**Panels A/B (concept schematic)**: NOT regenerated. These are
hand-designed Inkscape/Illustrator graphics committed as binary image
assets in the manuscript repo (``Manuscript/designed_figures/panels/fig6/
fig6_concept_ab.png``). They are not a CaliPPer reproduction artifact;
``generate_fig6_redesign.py`` outputs placeholder PNG/PDFs at the
fig6_a_placeholder and fig6_b_placeholder slots so panel assembly can
proceed without errors. For final figure assembly, substitute the
canonical PNG manually if needed.

**Stages that produce additional audit files (not in the verify gate)**:

  - ``reproduce_fig6_xbcr.py``     (XBCR fresh-inference variant; ΔAUROC
    +0.154, panel-exact per May 22 acceptance, NOT bit-exact because
    May-22 fresh-predictions file is permanently lost)

Invoke this audit script separately when needed:
    python reproduce/scripts/fig6/reproduce_fig6_xbcr.py

Usage::

    # Full pipeline in one command (most users want this):
    bash reproduce/reproduce_fig6.sh

    # Or via the main dispatcher:
    bash reproduce/reproduce.sh --figure 6

    # Direct invocation (if running outside the dispatcher):
    python reproduce/scripts/fig6.py \\
        --data-root reproduce/data/input \\
        --out-data  reproduce/data/output \\
        --out-fig   reproduce/figures/output/fig6

Flags:

  --skip-compute   reuse existing recal_summary_all.csv (skip Stage 1)
  --skip-render    skip Stage 3 (panel image rendering)
  --skip-panel-cd  skip Stage 2 (Panel C/D data computation)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

STUDY_ORDER = ["XBCR-net", "deepAntigen", "AntibioticsAI", "BigMHC", "PanPep"]
HERE = Path(__file__).resolve().parent


def run_subprocess(script_name: str, label: str, verbose: bool = True) -> int:
    """Invoke a script in reproduce/scripts/fig6/ as a subprocess.

    The compute scripts read inputs from INPUT_DIR via the shared _paths
    helper; they do NOT accept --data-root / --out-data arguments. Outputs
    go to OUTPUT_DIR (or FIG_DIR for renderer) unconditionally.
    """
    script = HERE / "fig6" / script_name
    if not script.exists():
        print(f"[fig6] ERROR: {script_name} not found at {script}")
        return 1
    cmd = [sys.executable, str(script)]
    print(f"[fig6] {label}: invoking {script_name}")
    proc = subprocess.run(cmd, capture_output=not verbose)
    if proc.returncode != 0:
        print(f"[fig6] ERROR: {script_name} exited with code {proc.returncode}")
        if not verbose and proc.stdout is not None:
            print(proc.stdout.decode())
        return proc.returncode
    return 0


def run_compute_recal(verbose: bool = True) -> int:
    return run_subprocess(
        "compute_fig6_recal_data.py", "Stage 1 (recal_data)", verbose)


def run_compute_panel_cd(verbose: bool = True) -> int:
    return run_subprocess(
        "compute_fig6_panel_c_d.py", "Stage 2 (Panel C/D data)", verbose)


def run_render_panels(verbose: bool = True) -> int:
    return run_subprocess(
        "generate_fig6_redesign.py", "Stage 3 (panel images c–p)", verbose)


def extract_canonical_values(summary_csv: Path) -> pd.DataFrame:
    """Pivot recal_summary_all.csv (long) to fig6_values.csv schema (wide)."""
    summary = pd.read_csv(summary_csv)
    rows = []
    for study in STUDY_ORDER:
        df = summary[summary["study"] == study]
        if df.empty:
            raise ValueError(f"Missing study in summary: {study}")
        aucroc_row = df[df["metric"] == "aucroc"].iloc[0]
        ap_row = df[df["metric"] == "ap"].iloc[0]
        rows.append({
            "study": study,
            "n": int(aucroc_row["n"]),
            "raw_aucroc": aucroc_row["before"],
            "cal_aucroc": aucroc_row["after"],
            "delta_aucroc": aucroc_row["delta"],
            "raw_ap": ap_row["before"],
            "cal_ap": ap_row["after"],
            "delta_ap": ap_row["delta"],
        })
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> int:
    out_data = Path(args.out_data)
    out_fig = Path(args.out_fig)
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    print(f"[fig6] data_root: {args.data_root}")
    print(f"[fig6] out_data:  {out_data}")
    print(f"[fig6] out_fig:   {out_fig}")

    summary_csv = out_data / "recal_data" / "recal_summary_all.csv"

    if not args.skip_compute or not summary_csv.exists():
        rc = run_compute_recal(verbose=True)
        if rc != 0:
            return rc
    else:
        print(f"[fig6] Stage 1: skip-compute set and {summary_csv.name} already exists; reusing")

    if not summary_csv.exists():
        print(f"[fig6] ERROR: {summary_csv} not produced by compute_fig6_recal_data.py")
        return 2

    if not args.skip_panel_cd:
        rc = run_compute_panel_cd(verbose=True)
        if rc != 0:
            print(f"[fig6] WARNING: Stage 2 (Panel C/D) failed; continuing without it")
    else:
        print(f"[fig6] Stage 2: skip-panel-cd set; skipping")

    if not args.skip_render:
        rc = run_render_panels(verbose=True)
        if rc != 0:
            print(f"[fig6] WARNING: Stage 3 (panel render) failed; continuing")
    else:
        print(f"[fig6] Stage 3: skip-render set; skipping")

    print(f"[fig6] Stage 4: extracting canonical fig6_values.csv from {summary_csv.name}")
    values = extract_canonical_values(summary_csv)
    out_csv = out_data / "fig6_values.csv"
    values.to_csv(out_csv, index=False, float_format="%.15g")
    print(f"[fig6] Wrote: {out_csv}")
    print("")
    print(values.to_string(index=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="Path to reproduce/data/input/ (committed to reproduce/data/input/)")
    parser.add_argument("--out-data", required=True,
                        help="Path to reproduce/data/output/ (where fig6_values.csv is written)")
    parser.add_argument("--out-fig", required=True,
                        help="Path to reproduce/figures/output/fig6/ (where rendered panels would go)")
    parser.add_argument("--skip-compute", action="store_true",
                        help="Skip Stage 1 (compute_fig6_recal_data.py); reuse existing recal_summary_all.csv if present")
    parser.add_argument("--skip-panel-cd", action="store_true",
                        help="Skip Stage 2 (compute_fig6_panel_c_d.py)")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip Stage 3 (generate_fig6_redesign.py panel rendering)")
    sys.exit(main(parser.parse_args()))

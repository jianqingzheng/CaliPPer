#!/usr/bin/env python3
"""Fig6 PRP-TCR (6th retrospective) — reproduce the recorded POSITIVE
CD69 recalibration result with the v2.7 API at the SINGLE optimal
configuration for PRP-TCR (no sweep).

Recorded result (investigation.md §24-PRP3, eval_cd69_recal_prp_sweep.py
best cell): canonical PRP recognition-domain distance, combine=sigma_C
(≡uniform — single varying peptide chain within-TCR), **θ=adaptive,
n_bins=4, anchor=off** →
    per-TCR mean ΔAUROC = +0.0198,  ΔAP = +0.0188,  ΣΔTDR@5 = +4
    (n=6 TDR-feasible TCRs; 1 degenerate = 135.1 reported ≡ raw, Δ0).
All three coupled metrics move UP TOGETHER (genuine-effect signature
per the coupled-metric-consistency rule; this cell is NOT one of the
ΣΔTDR>0-while-ΔAUROC<0 artifact cells). This script reproduces those
numbers EXACTLY and deposits the fig6 panel inputs.

v2.7 API — EXACT optimal PRP-TCR parameters, nothing else:
  calipper.core.fit_recalibration(cal_data, n_bins=4,
                                      threshold=None,        # adaptive θ
                                      train_anchor=None)     # anchor OFF
  calipper.core.apply_recalibration(..., n_bins=4, prev=cal_prev)
  - threshold=None  → v2.7 adaptive θ = max(2p−1, min(2p,0.5))
  - n_bins=4        → the PRP-TCR optimum (low-prevalence small-cal regime)
  - train_anchor=None → anchor OFF (the optimum; anchor never helped)
  - prev = cal_prev returned by fit_recalibration (NO test-label leak)
  NO other override is passed. n_bins is given to BOTH fit and apply
  (granularity match). This is the v2.7 default family specialised to
  the PRP-TCR optimum, not a hand-tuned deviation.

Distance: canonical S2DD `compute_combine_first_distances`
(sample → the TCR's training-POSITIVE recognition domain; Score==1,
train+valid, X-filtered). Effective chain = Epitope only: CDR3_b is
CONSTANT within a TCR (n_unique=1) so sigma_C provably zeroes it
(cd69_prp_sweep_weights.csv: Epitope=1.0, CDR3_b=0.0 for all 6 TCRs)
→ single peptide chain → sigma_C ≡ uniform (verified, Δ=0). Cal, test
and (unused here, anchor off) any reference distance all use this SAME
domain+weights (Invariant-2).

Invariant-1 CLEAN: cal = each TCR's OWN yeast TEST split predictions
(same model weights as the candidate predictions); candidates = the
SAME per-TCR model's CD69 scores (investigation_prediction_matrix_16x21
.csv). No retrain, no model mixing.

Degeneracy guard (the §24-FINAL fix, RETAINED): a recal that collapses
to a near-constant (saturated test_p; here 135.1) is flagged and
reported ≡ raw (ΔAUROC=ΔAP=0, top-k tie-broken by the raw score) so
coupled metrics cannot diverge by artifact.

SCOPE (§22-24, honest): the CD69 candidate pool is MODEL-SELECTED;
this is a hard, ceiling-limited, small-n (n=6) target. The gain is
small but GENUINE and coupled-consistent at the correct setting — it
is NOT manufactured structure. Reported as the 6th retrospective's
CD69 demonstration alongside the 5 main studies.

Outputs (panels/fig6/recal_data/):
  PRP-TCR_samples.csv       pooled rows for the panels:
        tcr_id, y_true, raw_pred, cal_pred, distance,
        raw_rank_in_tcr, cal_rank_in_tcr
  PRP-TCR_recal_per_tcr.csv per-TCR before/after AUROC/AP/top5 + degen
  PRP-TCR_summary.csv       headline (per-TCR mean ΔAUROC/ΔAP, ΣΔTDR@k,
                            pooled AUROC/AP) — reproduces §24-PRP3
"""
import glob
import importlib.util as _ilu
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..'))
PRP_DIR = os.path.join(PROJECT_ROOT, 'Model', 'PRP_TCR')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PRP_DIR)

from calipper.core import fit_recalibration, apply_recalibration  # noqa: E402
from calipper.combine_first_helpers import (  # noqa: E402
    compute_chain_weights, compute_combine_first_distances)

# Reuse the audited PRP-TCR primitives (load_split = Invariant-1-clean
# own-model split loader; safe_auc/ap; constants).
_spec = _ilu.spec_from_file_location(
    "recal", os.path.join(PRP_DIR, "eval_prp_tcr_recalibration.py"))
recal = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(recal)

OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'recal_data')
ACT = os.path.join(PRP_DIR, "data", "activation_binary.csv")
PRED = os.path.join(PRP_DIR, "investigation_prediction_matrix_16x21.csv")
ASDATA = os.path.join(PRP_DIR, "data", "ASdata-all")
PATIENT_TCRS = recal.PATIENT_TCRS
SEED = recal.SEED
LOGK, LOGB, K_TOPK = recal.LOGK, recal.LOGB, recal.K_TOPK
CAL_SUB = 6000
MIN_POS = 2
CHAIN_COLS = ["Epitope", "CDR3_b"]

# ── THE single optimal PRP-TCR configuration (§24-PRP3). Nothing else. ──
OPT_THRESHOLD = None                 # v2.7 adaptive θ
OPT_N_BINS = 4                       # PRP-TCR optimum
OPT_TRAIN_ANCHOR = None              # anchor OFF (optimum)
OPT_WEIGHT_FORMULA = "sigma_C"       # ≡ uniform (single peptide chain)
OPT_COMBINE_METHOD = "weighted_max_znorm"
TDR_KS = (1, 3, 5)


def load_prp_domain(t):
    """TCR's training-POSITIVE recognition domain (the PRP concept):
    Score==1, Split∈{train,valid}, X-filtered, set-deduped on Epitope.
    Identical rule to eval_cd69_recal_prp_sweep / eval_plan_b5."""
    df = pd.read_csv(glob.glob(os.path.join(ASDATA, f"{t}_neg=*.csv"))[0])
    df = df.loc[~df["Epitope"].astype(str).str.contains("X", na=False)]
    df = df.loc[df["Split"].isin(["train", "valid"]) & (df["Score"] == 1)]
    return (df[["Epitope", "CDR3_b"]].astype(str)
            .drop_duplicates("Epitope").reset_index(drop=True))


def effective_chains(domain_df):
    """Drop chains constant in the domain (n_unique==1) — their LogDist
    is a degenerate constant. Within-TCR this drops CDR3_b → ['Epitope']
    (= the sigma_C reduction made explicit; sigma_C weight CDR3_b=0)."""
    return [c for c in CHAIN_COLS if domain_df[c].nunique() > 1]


def prp_distance(sample_peps, cdr3b, domain_df, eff_cols, weights):
    sdf = pd.DataFrame({"Epitope": [str(p) for p in sample_peps],
                        "CDR3_b": cdr3b})
    K = min(K_TOPK, len(domain_df))
    return np.asarray(compute_combine_first_distances(
        sdf, domain_df, eff_cols, weights, k=LOGK, b=LOGB, K=K,
        combine_method=OPT_COMBINE_METHOD), float)


def rank_order(scores, raw):
    """Descending order; ties broken by the RAW model score so a
    non-informative recal cannot fabricate reorderings."""
    return np.lexsort((-np.asarray(raw, float), -np.asarray(scores, float)))


def is_degenerate(rc, tol=1e-6):
    return np.unique(np.round(np.asarray(rc, float), 6)).size <= 1 \
        or float(np.std(rc)) < tol


def ranks_in_tcr(scores, raw):
    """1-based rank of each candidate (1 = top), raw-tie-broken."""
    order = rank_order(scores, raw)
    r = np.empty(len(order), int)
    r[order] = np.arange(1, len(order) + 1)
    return r


def main():
    rng = np.random.default_rng(SEED)
    actb = pd.read_csv(ACT); actb["Peptide"] = actb["Peptide"].astype(str)
    predm = pd.read_csv(PRED).rename(columns={"Unnamed: 0": "Peptide"})
    predm["Peptide"] = predm["Peptide"].astype(str)

    samples, per_tcr = [], []
    for t in PATIENT_TCRS:
        if t not in actb.columns or t not in predm.columns:
            continue
        cd = actb[["Peptide", t]].merge(
            predm[["Peptide", t]], on="Peptide", suffixes=("_y", "_p"))
        cd = cd.dropna(subset=[f"{t}_p"]).reset_index(drop=True)
        y = cd[f"{t}_y"].astype(int).to_numpy()
        if y.sum() < MIN_POS:
            continue
        raw = cd[f"{t}_p"].astype(float).to_numpy()
        cand_pep = cd["Peptide"].tolist()

        dom = load_prp_domain(t)
        eff = effective_chains(dom)
        if not eff:
            print(f"[{t}] SKIP — domain has no varying chain")
            continue
        cdr3b = dom["CDR3_b"].iloc[0]
        w, _ = compute_chain_weights(dom, eff, LOGK, LOGB, K_TOPK,
                                     formula=OPT_WEIGHT_FORMULA)

        cand_d = prp_distance(cand_pep, cdr3b, dom, eff, w)
        cal_pep, cal_y, cal_p = recal.load_split(t, "test", rng, CAL_SUB)
        # RNG-SEQUENCE PARITY with the audited sweep
        # (eval_cd69_recal_prp_sweep.py): its preload draws load_split
        # 'test' THEN 'valid' per TCR off ONE shared rng. load_split's
        # stratified negative subsample consumes rng, so the 'valid'
        # draw shifts every subsequent TCR's cal composition (and the
        # degeneracy). anchor=off does not USE valid, but this draw is
        # REQUIRED so the recorded §24-PRP3 numbers reproduce EXACTLY
        # (faithful pipeline replication, not result-fitting).
        # ⚠ FRAGILE COUPLING: this mirrors eval_cd69_recal_prp_sweep.py's
        # per-TCR preload order (test→valid, shared rng, same guard
        # sequence). If that sweep's load order / skip logic ever
        # changes, this discard becomes wrong SILENTLY (no error) —
        # update this line to match the sweep if it is ever modified.
        recal.load_split(t, "valid", rng, CAL_SUB)
        cal_d = prp_distance(cal_pep, cdr3b, dom, eff, w)

        # ── v2.7 API @ the PRP-TCR optimum ONLY ──
        ppv, npv, p_pos, p_neg, cprev = fit_recalibration(
            {t: (cal_y, cal_p, cal_d)},
            n_bins=OPT_N_BINS, threshold=OPT_THRESHOLD,
            train_anchor=OPT_TRAIN_ANCHOR)
        rc = apply_recalibration(y, raw, cand_d, ppv, npv, p_pos, p_neg,
                                 n_bins=OPT_N_BINS, prev=cprev)

        degen = is_degenerate(rc)
        cal_pred = raw if degen else np.asarray(rc, float)   # ≡raw if degen
        rauc, cauc = recal.safe_auc(y, raw), recal.safe_auc(y, cal_pred)
        rap, cap = recal.safe_ap(y, raw), recal.safe_ap(y, cal_pred)
        r_rank = ranks_in_tcr(raw, raw)
        c_rank = ranks_in_tcr(cal_pred, raw)
        r_t5 = int(y[rank_order(raw, raw)[:5]].sum())
        c_t5 = int(y[rank_order(cal_pred, raw)[:5]].sum())

        per_tcr.append(dict(
            tcr_id=t, n_cand=len(y), n_act=int(y.sum()),
            degenerate=bool(degen),
            raw_auroc=round(rauc, 4), cal_auroc=round(cauc, 4),
            d_auroc=round(cauc - rauc, 4),
            raw_ap=round(rap, 4), cal_ap=round(cap, 4),
            d_ap=round(cap - rap, 4),
            # full-precision deltas — the per-TCR-mean summary/headline
            # must average THESE, not the 4dp-rounded columns above
            # (rounding-chain artifact otherwise stores ΔAP 0.0187 vs
            # the sweep's 0.0188; results-audit SHOULD-FIX 2026-05-18).
            # Dropped before the per_tcr CSV is written (human-clean).
            d_auroc_x=cauc - rauc, d_ap_x=cap - rap,
            raw_top5=r_t5, cal_top5=c_t5, d_top5=c_t5 - r_t5))
        for i in range(len(y)):
            samples.append(dict(
                tcr_id=t, y_true=int(y[i]),
                raw_pred=float(raw[i]), cal_pred=float(cal_pred[i]),
                distance=float(cand_d[i]),
                raw_rank_in_tcr=int(r_rank[i]),
                cal_rank_in_tcr=int(c_rank[i])))
        print(f"  {t:>7} act={int(y.sum())}/{len(y)} "
              f"{'[DEGEN→≡raw]' if degen else '':<13} "
              f"AUROC {rauc:.3f}->{cauc:.3f} ({cauc-rauc:+.4f}) "
              f"AP {rap:.3f}->{cap:.3f} ({cap-rap:+.4f}) "
              f"top5 {r_t5}->{c_t5} ({c_t5-r_t5:+d})")

    os.makedirs(OUT_DIR, exist_ok=True)
    sdf = pd.DataFrame(samples)
    pdf = pd.DataFrame(per_tcr)
    sdf.to_csv(os.path.join(OUT_DIR, "PRP-TCR_samples.csv"), index=False)

    # ── headline = §24-PRP3 (per-TCR mean ΔAUROC/ΔAP, ΣΔTDR@k) ──
    # averaged at FULL precision (d_*_x), then rounded once.
    n = len(pdf)
    mean_dauc = float(pdf.d_auroc_x.mean())
    mean_dap = float(pdf.d_ap_x.mean())
    pdf = pdf.drop(columns=["d_auroc_x", "d_ap_x"])   # keep CSV clean
    pdf.to_csv(os.path.join(OUT_DIR, "PRP-TCR_recal_per_tcr.csv"),
               index=False)
    pooled_y = sdf.y_true.to_numpy()
    pooled_rauc = recal.safe_auc(pooled_y, sdf.raw_pred.to_numpy())
    pooled_cauc = recal.safe_auc(pooled_y, sdf.cal_pred.to_numpy())
    pooled_rap = recal.safe_ap(pooled_y, sdf.raw_pred.to_numpy())
    pooled_cap = recal.safe_ap(pooled_y, sdf.cal_pred.to_numpy())
    rows = [
        dict(study="PRP-TCR", metric="aucroc", level="per_tcr_mean",
             before=round(float(pdf.raw_auroc.mean()), 4),
             after=round(float(pdf.cal_auroc.mean()), 4),
             delta=round(mean_dauc, 4), n=n),
        dict(study="PRP-TCR", metric="ap", level="per_tcr_mean",
             before=round(float(pdf.raw_ap.mean()), 4),
             after=round(float(pdf.cal_ap.mean()), 4),
             delta=round(mean_dap, 4), n=n),
        dict(study="PRP-TCR", metric="aucroc", level="pooled",
             before=round(pooled_rauc, 4), after=round(pooled_cauc, 4),
             delta=round(pooled_cauc - pooled_rauc, 4), n=len(sdf)),
        dict(study="PRP-TCR", metric="ap", level="pooled",
             before=round(pooled_rap, 4), after=round(pooled_cap, 4),
             delta=round(pooled_cap - pooled_rap, 4), n=len(sdf)),
    ]
    for k in TDR_KS:
        rr = sum(int(g.sort_values("raw_rank_in_tcr").head(k).y_true.sum())
                 for _, g in sdf.groupby("tcr_id"))
        cc = sum(int(g.sort_values("cal_rank_in_tcr").head(k).y_true.sum())
                 for _, g in sdf.groupby("tcr_id"))
        rows.append(dict(study="PRP-TCR", metric=f"tdr@{k}",
                         level="sum_over_tcr", before=rr, after=cc,
                         delta=cc - rr, n=n))
    smry = pd.DataFrame(rows)
    smry.to_csv(os.path.join(OUT_DIR, "PRP-TCR_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print(f"PRP-TCR CD69 recal @ optimal v2.7 config "
          f"(sigma_C≡uniform, θ=adaptive, n_bins=4, anchor=off), n={n}")
    print(f"  per-TCR mean ΔAUROC = {mean_dauc:+.4f}  "
          f"(§24-PRP3 recorded +0.0198)")
    print(f"  per-TCR mean ΔAP    = {mean_dap:+.4f}  "
          f"(§24-PRP3 recorded +0.0188)")
    sdt5 = int(pdf.d_top5.sum())
    print(f"  ΣΔTDR@5             = {sdt5:+d}  (§24-PRP3 recorded +4)")
    print(f"  degenerate (≡raw)   = "
          f"{pdf.loc[pdf.degenerate,'tcr_id'].tolist()}")
    print(f"  pooled AUROC {pooled_rauc:.3f}->{pooled_cauc:.3f}  "
          f"pooled AP {pooled_rap:.3f}->{pooled_cap:.3f}")
    ok = (abs(mean_dauc - 0.0198) < 5e-4 and abs(mean_dap - 0.0188) < 5e-4
          and sdt5 == 4)
    print(f"  REPRODUCES §24-PRP3: {'YES ✓' if ok else 'NO ✗ — investigate'}")
    print("=" * 70)
    print(f"Saved → {OUT_DIR}/PRP-TCR_{{samples,recal_per_tcr,summary}}.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fig 5 BCR CT: Bayesian recalibration (per-variant LOO).

Protocol (verified 2026-04-25):
  Model:    fold4 model (trained on fold0/1/2/3, 14,355 samples)
  Pool:     fold4 test (3,655) + A1-A11 (281) + unseen (1,256) + flu (1,226) = 6,418
  Domains:  SARS (4,074) / Flu (2,344) — separated, never mixed
  LOO:      Hold out one variant, fit PPV/NPV on ALL remaining (same domain),
            apply to held-out variant
  Output:   Pooled ΔAUROC per model across all held-out samples

  cal_mask = ~test_mask (ALL remaining, not just valid variants)
  fit_recalibration(cal_data) → apply_recalibration()
  CALIBRATION_LAM = 0.0
  Adaptive theta: max(2*prev-1, min(2*prev, 0.5))

Panels:
  - fig5_bcr_ct_dumbbell.pdf  (5-model ΔAUROC dumbbell)
  - fig5_bcr_ct_domain_bars.pdf (per-domain ΔAUROC bars)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.dirname(SCRIPT_DIR)
# Self-contained path anchors (BUILD_PLAN §1+§5.2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, FIG_DIR  # also adds CaliPPer/ to sys.path
from style_config import apply_publication_style, BCR_MODEL_COLORS, BCR_MODEL_DISPLAY
from calipper.general_evaluator import safe_metric
from calipper.core import fit_recalibration, apply_recalibration

apply_publication_style()

RESULTS = os.path.join(INPUT_DIR, 'results')
FOLD4CAL_DIR = os.path.join(RESULTS, 'bcr_bind_ct_fold4cal')

MODELS = ['xbcr', 'deepaai', 'mambaaai', 'mint', 'rleaai']
_STYLE_KEY = {'xbcr': 'xbcr_net', 'deepaai': 'deepaai', 'mambaaai': 'mambaaai',
              'mint': 'mint', 'rleaai': 'rleaai'}
MIN_SAMPLES = 30
PW, PH = 3.5, 3.0


def load_and_pool(model):
    cal = pd.read_csv(os.path.join(FOLD4CAL_DIR, model, 'cal_predictions.csv'))
    cal['source'] = 'fold4_test'
    parts = [cal]
    for ts in ['A1-A11', 'unseen', 'flu']:
        fp = os.path.join(FOLD4CAL_DIR, model, f'{ts}_predictions.csv')
        if not os.path.exists(fp):
            import sys as _s_bcr_pool
            print(f"  ⚠ MISSING [bcr-ct-recal]: model={model} ts={ts} not found at {fp}; skipping", file=_s_bcr_pool.stderr, flush=True)
            continue
        df = pd.read_csv(fp)
        df['source'] = ts
        if 'data_source' not in df.columns:
            df['data_source'] = 'flu' if ts == 'flu' else 'sars'
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def run_recalibration_loo(pooled):
    """Per-variant LOO recalibration within each domain.
    Returns dict: {domain: (all_y, all_raw, all_cal)}
    """
    domain_results = {}
    for domain in ['sars', 'flu']:
        domain_df = pooled[pooled['data_source'] == domain]
        variants = domain_df.groupby('variant_seq').size()
        valid = variants[variants >= MIN_SAMPLES].index.tolist()

        all_y, all_raw, all_cal = [], [], []
        for held_v in valid:
            test_mask = domain_df['variant_seq'] == held_v
            cal_mask = ~test_mask
            cal_sub = domain_df[cal_mask]; test_sub = domain_df[test_mask]
            if len(test_sub) < 10: continue
            test_y = test_sub['rbd'].values.astype(int)
            if test_y.sum() == 0 or test_y.sum() == len(test_y): continue
            cal_y = cal_sub['rbd'].values.astype(int)
            if cal_y.sum() < 3 or (len(cal_y) - cal_y.sum()) < 3: continue

            cal_data = {'cal': (cal_y,
                                cal_sub['pred_prob'].values.astype(float),
                                cal_sub['distance'].values.astype(float))}
            test_p = test_sub['pred_prob'].values.astype(float)
            test_d = test_sub['distance'].values.astype(float)

            ppv_p, npv_p, pp, pn, cal_prev = fit_recalibration(cal_data)
            cal_s = apply_recalibration(test_y, test_p, test_d, ppv_p, npv_p, pp, pn)

            all_y.extend(test_y.tolist())
            all_raw.extend(test_p.tolist())
            all_cal.extend(cal_s.tolist())

        domain_results[domain] = (np.array(all_y), np.array(all_raw), np.array(all_cal))
    return domain_results


# ═══════════════════════════════════════════
# Run recalibration for all models
# ═══════════════════════════════════════════
model_results = {}  # model → {'sars': (orig, recal), 'flu': (orig, recal), 'pooled': (orig, recal)}

for model in MODELS:
    display = BCR_MODEL_DISPLAY.get(_STYLE_KEY[model], model)
    print(f"\n=== {display} ===")
    pooled = load_and_pool(model)

    domain_res = run_recalibration_loo(pooled)

    model_results[model] = {}
    all_y_pool, all_raw_pool, all_cal_pool = [], [], []

    for domain in ['sars', 'flu']:
        y, raw, cal = domain_res[domain]
        if len(y) == 0: continue
        orig = safe_metric('aucroc', y, raw)
        recal = safe_metric('aucroc', y, cal)
        model_results[model][domain] = (orig, recal)
        all_y_pool.extend(y); all_raw_pool.extend(raw); all_cal_pool.extend(cal)
        print(f"  {domain.upper()}: {orig:.3f}→{recal:.3f} Δ={recal-orig:+.3f} (n={len(y)})")

    if all_y_pool:
        orig_p = safe_metric('aucroc', np.array(all_y_pool), np.array(all_raw_pool))
        recal_p = safe_metric('aucroc', np.array(all_y_pool), np.array(all_cal_pool))
        model_results[model]['pooled'] = (orig_p, recal_p)
        print(f"  Pooled: {orig_p:.3f}→{recal_p:.3f} Δ={recal_p-orig_p:+.3f} (n={len(all_y_pool)})")


# ═══════════════════════════════════════════
# Panel: 5-model dumbbell chart (pooled SARS+flu)
# ═══════════════════════════════════════════
print(f"\n--- Dumbbell panel ---")
fig, ax = plt.subplots(1, 1, figsize=(PW, PH))

sorted_models = sorted(model_results.items(),
                        key=lambda x: x[1].get('pooled', (0, 0))[1] - x[1].get('pooled', (0, 0))[0],
                        reverse=True)
yp = np.arange(len(sorted_models))[::-1]

for i, (model, res) in enumerate(sorted_models):
    if 'pooled' not in res: continue
    b, a = res['pooled']
    color = BCR_MODEL_COLORS[_STYLE_KEY[model]]
    display = BCR_MODEL_DISPLAY[_STYLE_KEY[model]]
    ax.plot([b, a], [yp[i], yp[i]], color=color, linewidth=2,
            solid_capstyle='round', alpha=0.5)
    ax.scatter(b, yp[i], color='white', edgecolor=color, s=25, zorder=5, linewidth=0.8)
    ax.scatter(a, yp[i], color=color, s=30, zorder=5, edgecolor='white', linewidth=0.4)
    d = a - b
    ax.text(max(b, a) + 0.01, yp[i], f'{d:+.3f}', va='center', fontsize=6,
            color=color, fontweight='bold')

ax.axvline(0.5, color='gray', linewidth=0.3, linestyle=':', alpha=0.5)
ax.set_yticks(yp)
ax.set_yticklabels([BCR_MODEL_DISPLAY[_STYLE_KEY[m]] for m, _ in sorted_models], fontsize=7)
ax.set_xlabel('AUROC', fontsize=8)
ax.set_title('BCR CT recalibration\n(per-variant LOO, fold4 model)', fontweight='bold', fontsize=9)

out = os.path.join(PANEL_DIR, 'fig5_bcr_ct_dumbbell')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: fig5_bcr_ct_dumbbell")


# ═══════════════════════════════════════════
# Panel: per-domain ΔAUROC grouped bars
# ═══════════════════════════════════════════
print(f"\n--- Domain bars panel ---")
fig, ax = plt.subplots(1, 1, figsize=(PW, PH))

DOMAIN_COLORS = {'sars': '#e74c3c', 'flu': '#3498db'}
x = np.arange(len(MODELS))
w = 0.35
for i, domain in enumerate(['sars', 'flu']):
    deltas = []
    for model in MODELS:
        if domain in model_results.get(model, {}):
            b, a = model_results[model][domain]
            deltas.append(a - b)
        else:
            deltas.append(0)
    offset = -w/2 + i * w
    bars = ax.bar(x + offset, deltas, w, label=domain.upper(),
                  color=DOMAIN_COLORS.get(domain, '#888'), alpha=0.8, edgecolor='white')
    for j, d in enumerate(deltas):
        if abs(d) > 0.001:
            ax.text(x[j] + offset, d + (0.003 if d >= 0 else -0.01),
                    f'{d:+.3f}', ha='center', fontsize=5, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([BCR_MODEL_DISPLAY[_STYLE_KEY[m]] for m in MODELS],
                     fontsize=7, rotation=30, ha='right')
ax.set_ylabel('ΔAUROC', fontsize=8)
ax.set_title('BCR CT recalibration by domain', fontweight='bold', fontsize=9)
ax.legend(fontsize=6, loc='upper right')

out = os.path.join(PANEL_DIR, 'fig5_bcr_ct_domain_bars')
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: fig5_bcr_ct_domain_bars")


print(f"\n=== Fig 5 BCR CT recalibration complete ===")

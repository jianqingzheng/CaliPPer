"""Regenerate the full XBCR-net Panel-1 calibration set (canonical ~1293 rows).

Optional regeneration script — not part of the standard reproduce_fig6 flow. The
reproduce pipeline consumes the committed cached artifact
`results/xbcr_retrospective/reproduction/panel1_cal_full_long_antigens.csv`
(+ `distance_cache_panel1_full.npz`); this script documents how that artifact was
produced and lets it be regenerated from the published XBCR-net model.

The default XBCR-net inference encodes each sequence into a [300,20] window and
applies vec_shift(seq_shift=20), so any antigen longer than 280 residues overflows
the window (20 + len > 300) and is dropped, leaving only the ~1003 short-antigen
rows. This script re-runs model_rbd_0 with long-antigen truncation so those rows are
retained, producing the full ~1293-row calibration set used for S2DD performance
prediction. (Recalibration is unaffected: it uses the 1003-row cal and is rank-based.)

REQUIREMENTS (external to this repo):
  - The published XBCR-net repo (model weights + example-experimental_data.xlsx):
      https://github.com/XBCR-net  (models/binding/binding-XBCR_net/model_rbd_0.tf)
  - A TensorFlow env (TF1-compat; runs on GPU). Point XBCR_ORIG_REPO at the checkout:
      XBCR_ORIG_REPO=/path/to/xbcr_original_repo python regenerate_xbcr_panel1_full_cal.py

Output: writes panel1_cal_full_long_antigens.csv into the reproduce results dir.
"""
import os
import sys
import numpy as np
import pandas as pd

# --- locate the external XBCR-net repo (model weights + experimental data) ---
XBCR_REPO = os.environ.get(
    "XBCR_ORIG_REPO",
    # default: bundled/co-located checkout next to this reproduce tree, if present
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 "..", "..", "data", "input", "Data",
                                 "retrospective_xbcr", "xbcr_original_repo")),
)
if not os.path.isdir(XBCR_REPO):
    sys.exit(f"XBCR-net repo not found at {XBCR_REPO}; set XBCR_ORIG_REPO. See header.")
os.chdir(XBCR_REPO)
sys.path.insert(0, ".")

import tensorflow.compat.v1 as tf  # noqa: E402
tf.disable_v2_behavior()
import networks                    # noqa: E402  (from the XBCR-net repo)
from utils import one_hot_encoder  # noqa: E402

np.random.seed(1)
SHIFT, MAXLEN = 20, 300
REPRO_RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "..", "..", "data", "input", "results",
                                             "xbcr_retrospective"))


def clean(s):
    s = str(s)
    for ch in (" ", "_", "\n", "\t"):
        s = s.replace(ch, "")
    return s


def encode(seq):
    """One-hot at [SHIFT : SHIFT+len]; long antigens truncated to fit the window."""
    seq = clean(seq)
    v = np.zeros([MAXLEN, 20], dtype=np.float32)
    use = min(len(seq), MAXLEN - SHIFT)   # <=280 kept as-is; >280 truncated
    if use > 0:
        v[SHIFT:SHIFT + use, :] = one_hot_encoder(seq[:use])
    return v


def main():
    ex = pd.read_excel("data/binding/exper/example-experimental_data.xlsx")
    test = ex[ex["TEST"] == 1].copy().reset_index(drop=True)
    for c in ["Heavy", "Light", "variant_seq"]:
        test[c] = test[c].fillna("").astype(str)

    def ok(r):
        h, l, a = clean(r["Heavy"]), clean(r["Light"]), clean(r["variant_seq"])
        return (len(h) > 10 and len(l) > 10 and len(a) > 10
                and h.isalpha() and l.isalpha() and a.isalpha())
    test = test[test.apply(ok, axis=1)].reset_index(drop=True)
    print(f"TEST rows to infer: {len(test)}")

    H = np.stack([encode(s) for s in test["Heavy"]])
    L = np.stack([encode(s) for s in test["Light"]])
    A = np.stack([encode(s) for s in test["variant_seq"]])

    net_core = networks.get_net("XBCR_net")
    sh = [300, 20]
    ih = tf.placeholder(tf.float32, [None, *sh])
    il = tf.placeholder(tf.float32, [None, *sh])
    ia = tf.placeholder(tf.float32, [None, *sh])
    pred_bind, _ = net_core([sh, sh, sh])([ih, il, ia])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(max_to_keep=1).restore(
        sess, "models/binding/binding-XBCR_net/model_rbd_0.tf")

    preds, BS = [], 128
    for i in range(0, len(test), BS):
        pb = sess.run(pred_bind, feed_dict={ih: H[i:i+BS], il: L[i:i+BS], ia: A[i:i+BS]})
        preds.append(np.asarray(pb).reshape(-1))
    sess.close()
    test["pred_prob"] = np.concatenate(preds)[:len(test)]

    keep = [c for c in ["Name", "Heavy", "Light", "variant_seq", "variant_name",
                        "rbd", "pred_prob"] if c in test.columns]
    out_csv = os.path.join(REPRO_RESULTS, "reproduction", "panel1_cal_full_long_antigens.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    test[keep].to_csv(out_csv, index=False)
    print(f"saved {len(test)} rows (antigen>280: {(test['variant_seq'].str.len()>280).sum()}) -> {out_csv}")


if __name__ == "__main__":
    main()

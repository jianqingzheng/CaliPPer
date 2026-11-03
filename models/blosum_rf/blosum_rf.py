# decompyle3 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0]
# Embedded file name: /home/data/Github/general_eval/Model/BLOSUM_RF/blosum_rf.py
# Compiled at: 2026-02-18 04:56:29
# Size of source mod 2**32: 8666 bytes
"""
BLOSUM62 + Random Forest baseline for TCR-epitope binding prediction.

Self-contained binary binding predictor using:
  - BLOSUM62 encoding of CDR3b (padded to max 20 AA) + peptide (padded to max 22 AA)
  - PCA to retain 90% variance
  - Random Forest classifier (n_estimators=300)

No external dependencies beyond scikit-learn, numpy, pandas.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
BLOSUM62 = {'A':[
  4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,0], 
 'R':[
  -1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,0], 
 'N':[
  -2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,0], 
 'D':[
  -2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,0], 
 'C':[
  0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,0], 
 'Q':[
  -1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,0], 
 'E':[
  -1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,0], 
 'G':[
  0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,0], 
 'H':[
  -2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,0], 
 'I':[
  -1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,0], 
 'L':[
  -1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,0], 
 'K':[
  -1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,0], 
 'M':[
  -1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,0], 
 'F':[
  -2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,0], 
 'P':[
  -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,0], 
 'S':[
  1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,0], 
 'T':[
  0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,0], 
 'W':[
  -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,0], 
 'Y':[
  -2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,0], 
 'V':[
  0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4,0], 
 'X':[
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
FEATURE_DIM = 21
MAX_LEN_TCR = 20
MAX_LEN_PEP = 22

def encode_sequence(seq, max_len):
    """Encode a single amino acid sequence using BLOSUM62 with zero-padding."""
    encoded = np.zeros((max_len, FEATURE_DIM))
    for i, aa in enumerate(seq[:max_len]):
        aa = aa.upper()
        if aa in BLOSUM62:
            encoded[i] = BLOSUM62[aa]
        else:
            encoded[i] = BLOSUM62["X"]

    return encoded.flatten()


def encode_pair(peptide, cdr3b):
    """Encode a peptide-CDR3b pair into a single feature vector."""
    pep_enc = encode_sequence(peptide, MAX_LEN_PEP)
    tcr_enc = encode_sequence(cdr3b, MAX_LEN_TCR)
    return np.concatenate([pep_enc, tcr_enc])


def encode_dataset(df, pep_col='peptide', tcr_col='CDR3b'):
    """Encode all peptide-CDR3b pairs in a DataFrame."""
    features = np.array([encode_pair(row[pep_col], row[tcr_col]) for _, row in df.iterrows()])
    return features


def train_and_predict(train_csv, test_csv, output_csv, pep_col='peptide', tcr_col='CDR3b', label_col='binder', n_estimators=300, pca_variance=0.9, random_state=42):
    """Train BLOSUM-RF on train_csv, predict on test_csv, save to output_csv.

    Input CSVs must have columns: peptide, CDR3b, binder (at minimum).
    Additional columns (CDR3a, etc.) are preserved in the output.

    Output CSV adds a 'prediction' column with binding probability [0, 1].
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    print(f"    BLOSUM-RF: encoding {len(train_df)} train + {len(test_df)} test samples...")
    X_train = encode_dataset(train_df, pep_col, tcr_col)
    X_test = encode_dataset(test_df, pep_col, tcr_col)
    y_train = train_df[label_col].astype(int).values
    model = Pipeline([
     (
      "pca", PCA(n_components=pca_variance, svd_solver="full")),
     (
      "rf",
      RandomForestClassifier(n_estimators=n_estimators,
        n_jobs=(-1),
        random_state=random_state,
        class_weight="balanced"))])
    model.fit(X_train, y_train)
    n_components = model.named_steps["pca"].n_components_
    print(f"    PCA retained {n_components} components ({pca_variance * 100:.0f}% variance)")
    probs = model.predict_proba(X_test)
    classes = model.named_steps["rf"].classes_
    if len(classes) == 2:
        pos_idx = list(classes).index(1)
        predictions = probs[:, pos_idx]
    elif len(classes) == 1:
        predictions = np.ones(len(X_test)) if classes[0] == 1 else np.zeros(len(X_test))
    else:
        predictions = probs[:, 1]
    out_df = test_df.copy()
    out_df["prediction"] = predictions
    out_df.to_csv(output_csv, index=False)
    print(f"    Saved {len(out_df)} predictions to {output_csv}")
    return (
     out_df, model)


def train_and_predict_multiple(train_csv, test_csvs, output_csvs, pep_col='peptide', tcr_col='CDR3b', label_col='binder', n_estimators=300, pca_variance=0.9, random_state=42):
    """Train BLOSUM-RF once on train_csv, predict on multiple test CSVs.

    Args:
        train_csv: Path to training CSV.
        test_csvs: Dict of {name: path} for test sets (e.g., {'val': ..., 'test': ...}).
        output_csvs: Dict of {name: path} for output CSVs (same keys as test_csvs).

    Returns:
        Dict of {name: (out_df, model)} for each test set.
    """
    train_df = pd.read_csv(train_csv)
    X_train = encode_dataset(train_df, pep_col, tcr_col)
    y_train = train_df[label_col].astype(int).values
    print(f"    BLOSUM-RF: encoding {len(train_df)} train samples...")
    model = Pipeline([
     (
      "pca", PCA(n_components=pca_variance, svd_solver="full")),
     (
      "rf",
      RandomForestClassifier(n_estimators=n_estimators,
        n_jobs=(-1),
        random_state=random_state,
        class_weight="balanced"))])
    model.fit(X_train, y_train)
    n_components = model.named_steps["pca"].n_components_
    print(f"    PCA retained {n_components} components ({pca_variance * 100:.0f}% variance)")
    classes = model.named_steps["rf"].classes_
    results = {}
    for name in test_csvs:
        test_df = pd.read_csv(test_csvs[name])
        X_test = encode_dataset(test_df, pep_col, tcr_col)
        print(f"    Predicting on {name}: {len(test_df)} samples...")
        probs = model.predict_proba(X_test)
        if len(classes) == 2:
            pos_idx = list(classes).index(1)
            predictions = probs[:, pos_idx]
        elif len(classes) == 1:
            predictions = np.ones(len(X_test)) if classes[0] == 1 else np.zeros(len(X_test))
        else:
            predictions = probs[:, 1]
        out_df = test_df.copy()
        out_df["prediction"] = predictions
        out_df.to_csv((output_csvs[name]), index=False)
        print(f"    Saved {len(out_df)} {name} predictions to {output_csvs[name]}")
        results[name] = (out_df, model)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BLOSUM62 + Random Forest TCR-epitope binding predictor")
    parser.add_argument("--train", required=True, help="Training CSV")
    parser.add_argument("--test", required=True, help="Test CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--pep-col", default="peptide")
    parser.add_argument("--tcr-col", default="CDR3b")
    parser.add_argument("--label-col", default="binder")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--pca-variance", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_predict((args.train),
      (args.test), (args.output), pep_col=(args.pep_col),
      tcr_col=(args.tcr_col),
      label_col=(args.label_col),
      n_estimators=(args.n_estimators),
      pca_variance=(args.pca_variance),
      random_state=(args.seed))

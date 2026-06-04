# decompyle3 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0]
# Embedded file name: Model/TCR_BERT/tcrbert_svm.py
# Compiled at: 2026-02-19 22:31:00
# Size of source mod 2**32: 11277 bytes
"""
TCR-BERT embedding + global SVM for TCR-epitope binding prediction.

Approach:
  1. Extract 768-dim mean-pooled embeddings from pre-trained TCR-BERT
     ("wukevin/tcr-bert") for CDR3b sequences
  2. Encode epitope with BLOSUM62 (padded to max 22 AA)
  3. Concatenate CDR3b embedding + epitope encoding
  4. PCA (50 components) + SVM(RBF, probability=True)

Uses only CDR3b + epitope (no CDR3a). Global SVM enables prediction on
unseen epitopes (unlike per-epitope SVMs in the benchmark notebook).

Input:  CSV with columns peptide, CDR3b, binder
Output: CSV with columns peptide, CDR3b, binder, prediction
"""
import os, sys, argparse, warnings, numpy as np, pandas as pd, joblib, torch
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
BLOSUM_DIM = 21
MAX_LEN_PEP = 22

def encode_epitope_blosum(seq, max_len=MAX_LEN_PEP):
    """Encode a single epitope sequence using BLOSUM62 with zero-padding."""
    encoded = np.zeros((max_len, BLOSUM_DIM))
    for i, aa in enumerate(seq[:max_len]):
        aa = aa.upper()
        encoded[i] = BLOSUM62.get(aa, BLOSUM62["X"])

    return encoded.flatten()


def encode_epitopes(epitopes, max_len=MAX_LEN_PEP):
    """Batch encode epitope sequences."""
    return np.array([encode_epitope_blosum(e, max_len) for e in epitopes])


def get_bert_embeddings(seqs, model_dir='wukevin/tcr-bert', layer=-1, method='mean', batch_size=256, device=0):
    """Extract TCR-BERT embeddings for CDR3b sequences.

    Reimplements the core of TCR-BERT's get_transformer_embeddings()
    without the complex dependency chain.
    """
    from transformers import BertModel, BertTokenizer
    if torch.cuda.is_available() and isinstance(device, int) and device >= 0:
        dev = torch.device(f"cuda:{device}")
    else:
        dev = torch.device("cpu")
    print(f"    Loading TCR-BERT from {model_dir}...")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir).to(dev)
    model.eval()
    ws_seqs = [" ".join(list(s)) for s in seqs]
    embeddings = []
    n_chunks = (len(ws_seqs) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(0, len(ws_seqs), batch_size):
            chunk = ws_seqs[i:i + batch_size]
            encoded = tokenizer(chunk,
              padding="max_length", max_length=64, truncation=True,
              return_tensors="pt")
            encoded = {k: v.to(dev) for k, v in encoded.items()}
            outputs = model(**encoded, **{"output_hidden_states": True})
            for j in range(len(chunk)):
                h = outputs.hidden_states[layer][j].cpu().numpy().astype(np.float64)
                seq_len = len(chunk[j].split())
                seq_hidden = h[1:1 + seq_len]
                if method == "mean":
                    embeddings.append(seq_hidden.mean(axis=0))
                elif method == "cls":
                    embeddings.append(h[0])
                else:
                    embeddings.append(seq_hidden.mean(axis=0))

            chunk_idx = i // batch_size + 1
            if not chunk_idx % 10 == 0:
                if chunk_idx == n_chunks:
                    pass
            print(f"      Embedded {min(i + batch_size, len(ws_seqs))}/{len(ws_seqs)} sequences")

    del model
    torch.cuda.empty_cache()
    return np.stack(embeddings)


def train_tcrbert_svm(train_csv, val_csv, model_path, n_pcs=50, bert_model='wukevin/tcr-bert', batch_size=256, device=0, seed=42):
    """Train TCR-BERT embedding + PCA + SVM pipeline.

    Saves: model artifacts (PCA, scaler, SVM) + precomputed embeddings.
    """
    np.random.seed(seed)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
    all_tcrs = pd.concat([train_df["CDR3b"], val_df["CDR3b"]]).tolist()
    all_peps = pd.concat([train_df["peptide"], val_df["peptide"]]).tolist()
    print("  Extracting TCR-BERT embeddings...")
    all_bert = get_bert_embeddings(all_tcrs, model_dir=bert_model, batch_size=batch_size,
      device=device)
    print(f"    Embedding shape: {all_bert.shape}")
    all_pep_enc = encode_epitopes(all_peps)
    print(f"    Epitope encoding shape: {all_pep_enc.shape}")
    all_features = np.hstack([all_bert, all_pep_enc])
    print(f"    Combined feature shape: {all_features.shape}")
    n_train = len(train_df)
    X_train = all_features[:n_train]
    X_val = all_features[n_train:]
    y_train = train_df["binder"].astype(int).values
    y_val = val_df["binder"].astype(int).values
    n_pcs_actual = min(n_pcs, X_train.shape[0], X_train.shape[1])
    n_nystroem = min(2000, X_train.shape[0] // 5)
    print(f"  Training PCA({n_pcs_actual}) + Nystroem({n_nystroem}) + LinearSVC...")
    base_pipeline = Pipeline([
     (
      "scaler", StandardScaler()),
     (
      "pca", PCA(n_components=n_pcs_actual)),
     (
      "nystroem",
      Nystroem(kernel="rbf", n_components=n_nystroem, random_state=seed)),
     (
      "svm",
      LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000))])
    pipeline = CalibratedClassifierCV(base_pipeline, cv=3)
    pipeline.fit(X_train, y_train)
    val_probs = pipeline.predict_proba(X_val)
    classes = pipeline.classes_
    if len(classes) == 2:
        pos_idx = list(classes).index(1)
        val_pred = val_probs[:, pos_idx]
    else:
        val_pred = val_probs[:, -1]
    from sklearn.metrics import roc_auc_score
    try:
        val_auc = roc_auc_score(y_val, val_pred)
    except ValueError:
        val_auc = 0.5
    else:
        print(f"  Val AUC: {val_auc:.4f}")
        joblib.dump(pipeline, model_path)
        print(f"  Saved model to {model_path}")
        return model_path


def predict_tcrbert_svm(model_path, test_csv, output_csv, bert_model='wukevin/tcr-bert', batch_size=256, device=0):
    """Load trained pipeline, extract embeddings, predict, save CSV."""
    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(test_csv)
    print(f"  Extracting TCR-BERT embeddings for {len(test_df)} test samples...")
    test_bert = get_bert_embeddings((test_df["CDR3b"].tolist()),
      model_dir=bert_model, batch_size=batch_size,
      device=device)
    test_pep_enc = encode_epitopes(test_df["peptide"].tolist())
    X_test = np.hstack([test_bert, test_pep_enc])
    probs = pipeline.predict_proba(X_test)
    classes = pipeline.classes_
    if len(classes) == 2:
        pos_idx = list(classes).index(1)
        predictions = probs[:, pos_idx]
    else:
        predictions = probs[:, -1]
    out_df = test_df.copy()
    out_df["prediction"] = predictions
    out_df.to_csv(output_csv, index=False)
    print(f"  Saved {len(out_df)} predictions to {output_csv}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCR-BERT + SVM for TCR-epitope binding prediction")
    parser.add_argument("--mode", required=True, choices=["train", "predict"])
    parser.add_argument("--train", type=str, help="Training CSV")
    parser.add_argument("--val", type=str, help="Validation CSV")
    parser.add_argument("--test", type=str, help="Test CSV (for predict mode)")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--output", type=str, help="Output CSV (for predict mode)")
    parser.add_argument("--bert-model", type=str, default="wukevin/tcr-bert", help="TCR-BERT model name or path")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for BERT embedding extraction")
    parser.add_argument("--device", type=int, default=0, help="GPU device index (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "train":
        if not (args.train and args.val):
            parser.error("--train and --val required for train mode")
        train_tcrbert_svm((args.train), (args.val), (args.model), n_pcs=(args.n_pcs),
          bert_model=(args.bert_model),
          batch_size=(args.batch_size),
          device=(args.device),
          seed=(args.seed))
    elif args.mode == "predict":
        if not (args.test and args.output):
            parser.error("--test and --output required for predict mode")
        predict_tcrbert_svm((args.model), (args.test), (args.output), bert_model=(args.bert_model),
          batch_size=(args.batch_size),
          device=(args.device))

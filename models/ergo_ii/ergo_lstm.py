# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0]
# Embedded file name: Model/ERGO_II/ergo_lstm.py
# Compiled at: 2026-02-19 12:29:05
# Size of source mod 2**32: 10655 bytes
"""
Standalone ERGO-II LSTM implementation for TCR-epitope binding prediction.

Architecture from ERGO-II (Springer et al., 2021):
  - LSTM_Encoder for CDR3b: Embedding(21,10) -> LSTM(10,500,2-layer) -> last cell
  - LSTM_Encoder for peptide: same architecture
  - MLP classifier: Linear(1000,31) -> LeakyReLU -> Dropout -> Linear(31,1) -> Sigmoid
  - BCELoss, Adam(lr=1e-4), 50 epochs

Uses only CDR3b + epitope (no CDR3a). Standard PyTorch training (no Lightning).

Input:  CSV with columns peptide, CDR3b, binder
Output: CSV with columns peptide, CDR3b, binder, prediction
"""
import os, sys, argparse, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["X"] = 0

def encode_sequence(seq):
    """Encode amino acid sequence as integer tensor."""
    return [AA_TO_IDX.get(aa.upper(), 0) for aa in seq]


class ERGODataset(Dataset):
    __doc__ = "PyTorch Dataset wrapping CSV data with peptide, CDR3b, binder columns."

    def __init__(self, df, pep_col='peptide', tcr_col='CDR3b', label_col='binder'):
        self.peptides = df[pep_col].tolist()
        self.tcrs = df[tcr_col].tolist()
        self.labels = df[label_col].astype(float).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pep = encode_sequence(self.peptides[idx])
        tcr = encode_sequence(self.tcrs[idx])
        label = self.labels[idx]
        return (pep, tcr, label)


def collate_fn(batch):
    """Pad variable-length sequences to batch max length."""
    peps, tcrs, labels = zip(*batch)
    pep_lens = torch.LongTensor([len(p) for p in peps])
    max_pep = max(pep_lens).item()
    pep_padded = torch.zeros((len(peps)), max_pep, dtype=(torch.long))
    for i, p in enumerate(peps):
        pep_padded[i, :len(p)] = torch.LongTensor(p)
    else:
        tcr_lens = torch.LongTensor([len(t) for t in tcrs])
        max_tcr = max(tcr_lens).item()
        tcr_padded = torch.zeros((len(tcrs)), max_tcr, dtype=(torch.long))
        for i, t in enumerate(tcrs):
            tcr_padded[i, :len(t)] = torch.LongTensor(t)
        else:
            labels = torch.FloatTensor(labels)
            return (pep_padded, pep_lens, tcr_padded, tcr_lens, labels)


class LSTMEncoder(nn.Module):
    __doc__ = "LSTM encoder for amino acid sequences (from ERGO-II Models.py)."

    def __init__(self, embedding_dim=10, lstm_dim=500, dropout=0.1):
        super().__init__()
        self.lstm_dim = lstm_dim
        self.embedding = nn.Embedding(21, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True,
          dropout=dropout)

    def init_hidden(self, batch_size, device):
        return (
         autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim).to(device)),
         autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim).to(device)))

    def forward(self, seq, lengths):
        device = seq.device
        embeds = self.embedding(seq)
        lengths, perm_idx = lengths.sort(0, descending=True)
        embeds = embeds[perm_idx]
        packed = nn.utils.rnn.pack_padded_sequence(embeds, (lengths.cpu()), batch_first=True)
        hidden = self.init_hidden(len(lengths), device)
        self.lstm.flatten_parameters()
        lstm_out, hidden = self.lstm(packed, hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        last_cell = torch.cat([lstm_out[(i, lengths[i] - 1)].unsqueeze(0) for i in range(len(lengths))],
          dim=0)
        return last_cell


class ERGO_LSTM(nn.Module):
    __doc__ = "ERGO-II model: dual LSTM encoders + MLP classifier."

    def __init__(self, embedding_dim=10, lstm_dim=500, dropout=0.1):
        super().__init__()
        self.tcr_encoder = LSTMEncoder(embedding_dim, lstm_dim, dropout)
        self.pep_encoder = LSTMEncoder(embedding_dim, lstm_dim, dropout)
        mlp_dim = lstm_dim * 2
        hidden_dim = int(np.sqrt(mlp_dim))
        self.hidden_layer = nn.Linear(mlp_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, pep_seq, pep_lens, tcr_seq, tcr_lens):
        tcr_enc = self.tcr_encoder(tcr_seq, tcr_lens)
        pep_enc = self.pep_encoder(pep_seq, pep_lens)
        concat = torch.cat([tcr_enc, pep_enc], dim=1)
        hidden = self.drop(self.relu(self.hidden_layer(concat)))
        output = torch.sigmoid(self.output_layer(hidden))
        return output.squeeze(1)


def train_ergo(train_csv, val_csv, model_path, epochs=50, batch_size=64, lr=0.0001, patience=5, seed=42):
    """Train ERGO-II LSTM with early stopping on validation AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
    train_loader = DataLoader((ERGODataset(train_df)), batch_size=batch_size, shuffle=True,
      collate_fn=collate_fn,
      num_workers=0,
      drop_last=False)
    val_loader = DataLoader((ERGODataset(val_df)), batch_size=batch_size, shuffle=False,
      collate_fn=collate_fn,
      num_workers=0,
      drop_last=False)
    model = ERGO_LSTM().to(device)
    optimizer = torch.optim.Adam((model.parameters()), lr=lr)
    criterion = nn.BCELoss()
    best_auc = 0.0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

    for pep, pep_len, tcr, tcr_len, labels in train_loader:
        pep, pep_len = pep.to(device), pep_len.to(device)
        tcr, tcr_len = tcr.to(device), tcr_len.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(pep, pep_len, tcr, tcr_len)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    else:
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for pep, pep_len, tcr, tcr_len, labels in val_loader:
                pep, pep_len = pep.to(device), pep_len.to(device)
                tcr, tcr_len = tcr.to(device), tcr_len.to(device)
                outputs = model(pep, pep_len, tcr, tcr_len)
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.5
        else:
            if not (epoch + 1) % 5 == 0:
                if epoch == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}: loss={train_loss / n_batches:.4f}, val_auc={val_auc:.4f}")
                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0
                    torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1} (best val_auc={best_auc:.4f})")
                    break
                print(f"  Best val AUC: {best_auc:.4f}")
                return model_path


def predict_ergo(model_path, test_csv, output_csv, batch_size=64):
    """Load trained ERGO-II model, predict on test set, save CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(test_csv)
    test_loader = DataLoader((ERGODataset(test_df)), batch_size=batch_size, shuffle=False,
      collate_fn=collate_fn,
      num_workers=0,
      drop_last=False)
    model = ERGO_LSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for pep, pep_len, tcr, tcr_len, labels in test_loader:
            pep, pep_len = pep.to(device), pep_len.to(device)
            tcr, tcr_len = tcr.to(device), tcr_len.to(device)
            outputs = model(pep, pep_len, tcr, tcr_len)
            all_probs.extend(outputs.cpu().numpy())

    out_df = test_df.copy()
    out_df["prediction"] = all_probs
    out_df.to_csv(output_csv, index=False)
    print(f"  Saved {len(out_df)} predictions to {output_csv}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERGO-II LSTM for TCR-epitope binding prediction")
    parser.add_argument("--mode", required=True, choices=["train", "predict"])
    parser.add_argument("--train", type=str, help="Training CSV")
    parser.add_argument("--val", type=str, help="Validation CSV")
    parser.add_argument("--test", type=str, help="Test CSV (for predict mode)")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--output", type=str, help="Output CSV (for predict mode)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "train":
        if args.train:
            if not args.val:
                parser.error("--train and --val required for train mode")
            train_ergo((args.train), (args.val), (args.model), epochs=(args.epochs),
              batch_size=(args.batch_size),
              lr=(args.lr),
              patience=(args.patience),
              seed=(args.seed))
        else:
            pass
    if args.mode == "predict":
        if not (args.test and args.output):
            parser.error("--test and --output required for predict mode")
        predict_ergo((args.model), (args.test), (args.output), batch_size=(args.batch_size))

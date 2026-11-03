"""Shared style constants for manuscript figures."""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Manuscript" / "Overleaf" / "figures"

# --- Model styles ---
MODEL_COLORS = {
    "nettcr": "#1f77b4", "atm_tcr": "#ff7f0e", "blosum_rf": "#2ca02c",
    "ergo_ii": "#d62728", "tcrbert": "#9467bd",
}
MODEL_MARKERS = {
    "nettcr": "o", "atm_tcr": "s", "blosum_rf": "^",
    "ergo_ii": "D", "tcrbert": "v",
}
MODEL_DISPLAY = {
    "nettcr": "NetTCR", "atm_tcr": "ATM-TCR", "blosum_rf": "BLOSUM-RF",
    "ergo_ii": "ERGO-II", "tcrbert": "TCR-BERT",
}

# --- BCR model styles ---
BCR_MODEL_COLORS = {
    "xbcr_net": "#1f77b4", "deepaai": "#ff7f0e", "mambaaai": "#2ca02c",
    "mint": "#d62728", "rleaai": "#9467bd",
}
BCR_MODEL_MARKERS = {
    "xbcr_net": "o", "deepaai": "s", "mambaaai": "^",
    "mint": "D", "rleaai": "v",
}
BCR_MODEL_DISPLAY = {
    "xbcr_net": "XBCR-net", "deepaai": "DeepAAI", "mambaaai": "MambaAAI",
    "mint": "MINT", "rleaai": "RLEAAI",
}

# --- Source styles (for BCR test sources) ---
SOURCE_COLORS = {
    "xbcr_train": "#3498db", "A1-A11": "#e74c3c", "unseen": "#f39c12",
    "guoyu": "#2ecc71", "BNT162b2": "#9b59b6", "flu": "#e67e22",
}
SOURCE_DISPLAY = {
    "xbcr_train": "Training (SARS)", "A1-A11": "A1-A11",
    "unseen": "Unseen variants", "guoyu": "Guoyu",
    "BNT162b2": "BNT162b2", "flu": "Influenza",
}

# --- Fold styles (for per-fold degradation curves) ---
FOLD_COLORS = {
    0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd",
}
FOLD_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}
FOLD_DISPLAY = {i: f"Fold {i}" for i in range(5)}

# --- Metric styles ---
METRIC_COLORS = {
    "aucroc": "#3498db", "ap": "#2ecc71", "f1": "#e74c3c",
    "mcc": "#9b59b6", "brier": "#e67e22", "bss": "#1abc9c",
}
METRIC_MARKERS = {
    "aucroc": "o", "ap": "s", "f1": "^",
    "mcc": "D", "brier": "v", "bss": "P",
}
METRIC_DISPLAY = {
    "aucroc": "AUCROC", "ap": "AP", "f1": "F1",
    "mcc": "MCC", "brier": "Brier", "bss": "BSS",
}

# --- Publication settings ---
DPI = 300
FONT_LABEL = 14
FONT_TICK = 11
FONT_TITLE = 14
FONT_LEGEND = 11
GRID_ALPHA = 0.15

COL_SINGLE = 3.5  # inches
COL_DOUBLE = 7.0  # inches


def apply_publication_style():
    """Set matplotlib rcParams for Nature Methods publication style."""
    mpl.rcParams.update({
        # Typography: Arial/Helvetica, regular weight labels
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_TICK,
        "axes.labelsize": FONT_LABEL,
        "axes.labelweight": "normal",  # Nature: regular weight axis labels
        "axes.titlesize": FONT_TITLE,
        "axes.titleweight": "bold",
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "legend.frameon": True,         # Keep frame for readability
        "legend.framealpha": 0.85,      # Semi-transparent white background
        "legend.edgecolor": "none",     # Nature: no visible border
        "legend.borderpad": 0.3,
        # Spines: remove top/right (Nature standard)
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks: outward direction
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # Grid: off by default (Nature standard)
        "axes.grid": False,
        # Output
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })

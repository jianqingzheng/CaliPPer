"""CaliPPer — Calibration and Prediction of Performance for binding-prediction models.

Quick-start API (all imported here for convenience):

    from calipper import (
        # Distance computation
        compute_s2dd_distances,
        # Performance prediction
        predict_metric,
        predict_subset_metric,
        # Bayesian recalibration
        fit_recalibration,
        apply_recalibration,
        # Curve fitting (advanced)
        fit_best_curve,
        predict_best_curve,
        adaptive_n_bins,
        # Main evaluator class
        General_Evaluator,
    )

Two pipelines with SEPARATE regularisation lambdas — never mix them:

  Performance prediction (predict_metric, predict_subset_metric):
    - VBIAS_BETA_LAM = 0.05 — L2 on beta only in joint curve+beta fitting
    - Per-set binning, adaptive n_bins = max(4, min(8, n_minority // 8))
    - Dual curve fitting (exp + right-Gaussian) with parsimony delta > 0.02

  Bayesian recalibration (fit_recalibration, apply_recalibration):
    - CALIBRATION_LAM = 0.0 — no regularisation on PPV/NPV curve beta
    - ALWAYS use adaptive theta: threshold = max(2*prev-1, min(2*prev, 0.5))
    - Calibration source is study-specific (see docs/methods_summary.md)

See `docs/methods_summary.md` and `docs/api_reference.md` for full documentation.
"""

__version__ = "0.1.0"
__author__ = "CaliPPer Authors"
__license__ = "Apache-2.0"

# Canonical v2.7 public API
from .core import (  # noqa: F401
    # Distance computation
    compute_s2dd_distances,
    # Performance prediction
    predict_metric,
    predict_subset_metric,
    # Bayesian recalibration
    fit_recalibration,
    apply_recalibration,
    # Dual curve fitting
    fit_best_curve,
    predict_best_curve,
    fit_right_gaussian,
    predict_right_gaussian,
    # Adaptive bins
    adaptive_n_bins,
    # Constants
    VBIAS_BETA_LAM,
)

# Main evaluator class + utilities
from .general_evaluator import General_Evaluator  # noqa: F401
from .utils import read_table, column_filter, df_to_markdown  # noqa: F401

__all__ = [
    "__version__",
    "compute_s2dd_distances",
    "predict_metric",
    "predict_subset_metric",
    "fit_recalibration",
    "apply_recalibration",
    "fit_best_curve",
    "predict_best_curve",
    "fit_right_gaussian",
    "predict_right_gaussian",
    "adaptive_n_bins",
    "VBIAS_BETA_LAM",
    "General_Evaluator",
    "read_table",
    "column_filter",
    "df_to_markdown",
]

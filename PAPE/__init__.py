"""PAPE: Probabilistic Adaptive Performance Estimation.

Reference: Białek, Kivimäki, Kuberski, Perrakis. "Estimating Model Performance
Under Covariate Shift Without Labels." arXiv:2401.08348 (NeurIPS 2025).
"""
from .pape_core import pape_estimate

__all__ = ['pape_estimate']

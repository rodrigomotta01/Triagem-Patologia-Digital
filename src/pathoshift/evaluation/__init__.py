"""
Módulo de avaliação quantitativa e qualitativa do pipeline.
"""

from .metrics import compute_metrics, plot_confusion_matrix, save_metrics
from .error_analysis import find_errors, plot_error_examples

__all__ = [
    "compute_metrics",
    "plot_confusion_matrix",
    "save_metrics",
    "find_errors",
    "plot_error_examples"
]


"""
Módulo de extração de características (features) de imagens histopatológicas.
"""

from .extractor import (
    extract_color_histogram,
    extract_lbp_features,
    extract_global_stats,
    extract_all_features
)

__all__ = [
    "extract_color_histogram",
    "extract_lbp_features",
    "extract_global_stats",
    "extract_all_features"
]


"""
Módulo de pré-processamento de imagens: controle de qualidade e normalização de cor.
"""

from .quality import detect_background, assess_focus, quality_report
from .color_norm import ReinhardNormalizer

__all__ = [
    "detect_background",
    "assess_focus",
    "quality_report",
    "ReinhardNormalizer"
]


"""
quality.py — Controle de qualidade e detecção de artefatos em imagens microscópicas.

PROBLEMAS RECORRENTES EM HISTOPATOLOGIA DIGITAL:
1. TILES COM FUNDO BRANCO: Regiões vazias da lamínula de vidro sem tecido biológico.
   Se incluídos indiscriminadamente no classificador, introduzem ruído espúrio e descalibram
   as probabilidades.
   Solução: Segmentar a máscara de pixels quase brancos (>220 no canal de luminância).

2. DESFOQUE DE AQUISIÇÃO (OUT-OF-FOCUS):
   O scanner óptico ou microscópio pode perder a distância focal adequada na lâmina.
   Solução: Operador Laplaciano (segunda derivada espacial ∇²f = ∂²f/∂x² + ∂²f/∂y²).
   Bordas celulares nítidas geram altas oscilações de derivada (alta variância).
   Áreas borradas suavizam as transições (baixa variância).
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2


def detect_background(
    image: np.ndarray,
    threshold: float = 0.85,
    white_pixel_val: int = 220
) -> Tuple[bool, float]:
    """
    Detecta se uma imagem é predominantemente fundo (vidro sem tecido).

    Args:
        image: Array RGB (H, W, 3) uint8.
        threshold: Fração de área mínima considerada fundo (ex: 0.85 = 85%).
        white_pixel_val: Limiar de intensidade (0-255) em escala de cinza.

    Returns:
        (is_background, white_ratio)
    """
    if image is None or image.size == 0:
        return True, 1.0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    white_pixels = np.sum(gray >= white_pixel_val)
    white_ratio = float(white_pixels) / float(gray.size)

    return white_ratio >= threshold, white_ratio


def assess_focus(
    image: np.ndarray,
    threshold: float = 100.0
) -> Tuple[bool, float]:
    """
    Avalia a nitidez do foco utilizando a variância do operador diferencial Laplaciano.

    Args:
        image: Array RGB (H, W, 3) uint8.
        threshold: Limiar de corte empírico (abaixo deste valor = desfocado).

    Returns:
        (is_focused, laplacian_variance)
    """
    if image is None or image.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # CV_64F preserva números negativos e amplitudes antes de calcular a variância
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    return variance >= threshold, variance


def quality_report(image: np.ndarray) -> Dict[str, Any]:
    """
    Gera um relatório descritivo completo com métricas de qualidade para inspeção.
    """
    is_bg, bg_ratio = detect_background(image)
    is_foc, foc_var = assess_focus(image)

    mean_rgb = image.mean(axis=(0, 1)).round(2).tolist()
    std_rgb = image.std(axis=(0, 1)).round(2).tolist()

    return {
        "shape": list(image.shape),
        "is_background": is_bg,
        "background_ratio": round(bg_ratio, 4),
        "is_in_focus": is_foc,
        "focus_variance": round(foc_var, 2),
        "mean_rgb": mean_rgb,
        "std_rgb": std_rgb
    }


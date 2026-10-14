"""
color_norm.py — Normalização de coloração histopatológica (Método de Reinhard et al., 2001).

POR QUE A NORMALIZAÇÃO DE COR É FUNDAMENTAL EM PATOLOGIA DIGITAL?
As lâminas histopatológicas coradas com Hematoxilina e Eosina (H&E) sofrem grande variabilidade:
- Concentração do lote e envelhecimento do reagente químico.
- Diferença no tempo de exposição da amostra no protocolo de coloração.
- Diferenças espectrais nos sensores ópticos e fontes de luz dos scanners (ex: Leica, Hamamatsu, Philips).

MÉTODO DE REINHARD:
1. Converte a imagem do espaço RGB para o espaço perceptualmente decorrelacionado L*a*b*:
   - L*: Canal de Luminância / Brilho (0 a 100).
   - a*: Eixo verde (-) até vermelho (+).
   - b*: Eixo azul (-) até amarelo (+).
2. Computa a média (μ) e desvio padrão (σ) de cada canal na imagem de REFERÊNCIA (alvo padronizado).
3. Para cada nova imagem de entrada (fonte):
   I_norm[c] = (I_source[c] - μ_source[c]) * (σ_target[c] / σ_source[c]) + μ_target[c]
4. Reconverte a imagem normalizada do espaço L*a*b* de volta para RGB.
"""

from typing import Optional, Union
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


class ReinhardNormalizer:
    """
    Normalizador de cor baseado em estatísticas de média e desvio padrão no espaço L*a*b*.
    """

    def __init__(self) -> None:
        self.target_mean: Optional[np.ndarray] = None
        self.target_std: Optional[np.ndarray] = None

    def fit(self, target_image: Union[np.ndarray, str, Path]) -> "ReinhardNormalizer":
        """
        Calcula as estatísticas de cor a partir da imagem de referência.

        Args:
            target_image: Array NumPy RGB (H, W, 3) ou caminho de arquivo.
        """
        if isinstance(target_image, (str, Path)):
            with Image.open(target_image) as img:
                target_image = np.array(img.convert("RGB"), dtype=np.uint8)

        if target_image.ndim != 3 or target_image.shape[2] != 3:
            raise ValueError("A imagem de referência deve ser RGB com 3 canais.")

        lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Médias e desvios ao longo dos eixos espaciais (H, W)
        self.target_mean = lab.mean(axis=(0, 1))
        self.target_std = lab.std(axis=(0, 1))

        # Evita divisão por zero caso a imagem de referência tenha canal flat
        self.target_std = np.where(self.target_std < 1e-6, 1.0, self.target_std)
        return self

    def transform(self, source_image: np.ndarray) -> np.ndarray:
        """
        Aplica a transferência estatística de cor na imagem fornecida.

        Args:
            source_image: Array NumPy RGB (H, W, 3) uint8.

        Returns:
            Imagem normalizada no formato RGB uint8.
        """
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError("Execute fit(reference_image) antes de chamar transform().")

        if source_image.ndim != 3 or source_image.shape[2] != 3:
            return source_image

        lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB).astype(np.float32)

        src_mean = lab.mean(axis=(0, 1))
        src_std = lab.std(axis=(0, 1))
        src_std = np.where(src_std < 1e-6, 1.0, src_std)

        # Ajuste canal a canal
        for i in range(3):
            lab[:, :, i] = (
                (lab[:, :, i] - src_mean[i]) * (self.target_std[i] / src_std[i])
            ) + self.target_mean[i]

        # Trunca valores para o intervalo válido [0, 255]
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


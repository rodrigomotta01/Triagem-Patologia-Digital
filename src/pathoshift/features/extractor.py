"""
extractor.py — Extração de características (features) artesanais para classificação clássica.

POR QUE EXTRAÇÃO DE FEATURES MANUAIS?
Em vez de depender de uma rede neural 'caixa-preta', o processamento clássico de imagens
extrai descritores matemáticos interpretáveis diretamente dos pixels. Isso nos permite entender
e justificar cientificamente cada parâmetro durante a defesa acadêmica.

COMPOSIÇÃO DO VETOR DE FEATURES (129 DIMENSÕES):
1. HISTOGRAMA DE COR HSV (96 valores):
   - Converte de RGB para HSV (Hue, Saturation, Value).
   - O canal Hue (Matiz) mapeia o comprimento de onda da cor (roxo basofílico da Hematoxilina,
     rosa eosinofílico da Eosina).
   - O canal Saturation (Saturação) discrimina tecido denso de áreas claras/transparentes.
   - O canal Value (Brilho) mede a transmitância da luz microscópica.
   - 32 bins por canal * 3 canais = 96 dimensões (normalizadas por soma unitária).

2. TEXTURA POR LOCAL BINARY PATTERN - LBP (26 valores):
   - Avalia a microtextura e arranjo celular comparando a intensidade de cada pixel central
     com seus P vizinhos dispostos em um raio R (R=3, P=24).
   - Utiliza padrões uniformes (method='uniform'), agrupando rotações e reduzindo ruído.
   - Histograma de 26 bins normalizado.

3. ESTATÍSTICAS GLOBAIS E ENTROPIA (7 valores):
   - Média e desvio padrão para cada canal R, G, B (6 valores).
   - Entropia de Shannon da imagem em escala de cinza (1 valor):
     H = -sum(p(x) * log2(p(x)))
     Mede a complexidade/desordem do arranjo tecidual.
"""

from typing import Optional
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import entropy


def extract_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Extrai o histograma de cores no espaço HSV.
    
    Args:
        image: Array RGB uint8 (H, W, 3).
        bins: Quantidade de intervalos por canal.

    Returns:
        Vetor 1D concatenado de tamanho (3 * bins).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    features = []

    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256]).flatten()
        # Normalização L1 (probabilidade)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        features.append(hist)

    return np.concatenate(features).astype(np.float32)


def extract_lbp_features(
    image: np.ndarray,
    radius: int = 3,
    n_points: int = 24,
    bins: int = 26
) -> np.ndarray:
    """
    Extrai descritores de microtextura utilizando Local Binary Pattern (LBP uniforme).

    Args:
        image: Array RGB uint8.
        radius: Raio da vizinhança circular.
        n_points: Número de pontos amostrados no círculo.
        bins: Bins do histograma LBP (para uniforme, tipicamente n_points + 2).

    Returns:
        Histograma 1D normalizado com densidade de probabilidade.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method="uniform")

    # Padrões uniformes geram valores de 0 até (n_points + 1)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=bins,
        range=(0, n_points + 2),
        density=True
    )
    return hist.astype(np.float32)


def extract_global_stats(image: np.ndarray) -> np.ndarray:
    """
    Calcula médias, desvios padrões por canal de cor e a entropia de Shannon de intensidade.

    Returns:
        Array 1D com 7 elementos numéricos.
    """
    stats = []

    # 1. Média e Desvio Padrão RGB (6 valores)
    for c in range(3):
        channel_data = image[:, :, c].astype(np.float32)
        stats.append(float(channel_data.mean()))
        stats.append(float(channel_data.std()))

    # 2. Entropia de Shannon em escala de cinza (1 valor)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist_gray, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    # Adiciona epsilon para evitar log(0)
    shannon_entropy = float(entropy(hist_gray + 1e-12, base=2))
    stats.append(shannon_entropy)

    return np.array(stats, dtype=np.float32)


def extract_all_features(
    image: np.ndarray,
    hsv_bins: int = 32,
    lbp_radius: int = 3,
    lbp_n_points: int = 24,
    lbp_bins: int = 26
) -> np.ndarray:
    """
    Gera o vetor completo unificado de 129 características para a imagem.
    """
    f_color = extract_color_histogram(image, bins=hsv_bins)
    f_texture = extract_lbp_features(
        image, radius=lbp_radius, n_points=lbp_n_points, bins=lbp_bins
    )
    f_stats = extract_global_stats(image)

    return np.concatenate([f_color, f_texture, f_stats]).astype(np.float32)


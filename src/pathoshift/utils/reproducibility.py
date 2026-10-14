"""
reproducibility.py — Garante que experimentos sejam 100% reproduzíveis.

POR QUE FIXAR SEEDS?
Muitas operações computacionais utilizam números pseudoaleatórios (particionamento estratificado,
inicialização de otimizadores, embaralhamento). Sem semente fixa, repetições do script gerariam
resultados ligeiramente diferentes, prejudicando a comparabilidade rigorosa de hipóteses.

POR QUE REGISTRAR INFORMAÇÕES DE AMBIENTE?
Mudanças em versões de bibliotecas (scikit-learn, OpenCV) podem alterar detalhes de implementação
de algoritmos. O enunciado exige registrar ambiente, seeds e configurações.
"""

import random
import sys
import platform
from datetime import datetime
from typing import Dict, Any
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Fixa todas as sementes de aleatoriedade nos módulos relevantes.
    
    Args:
        seed: Inteiro para a semente pseudoaleatória.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_environment_info() -> Dict[str, Any]:
    """
    Captura e retorna um dicionário contendo versões do SO, Python e principais pacotes.
    """
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        info["sklearn_version"] = "not_installed"

    try:
        import cv2
        info["opencv_version"] = cv2.__version__
    except ImportError:
        info["opencv_version"] = "not_installed"

    try:
        import skimage
        info["skimage_version"] = skimage.__version__
    except ImportError:
        info["skimage_version"] = "not_installed"

    return info


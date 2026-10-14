"""
error_analysis.py — Análise qualitativa e visualização de casos de erro.

POR QUE A ANÁLISE DE ERROS É ESSENCIAL?
No edital do PathoShift Challenge, métricas numéricas isoladas não são suficientes.
O grupo deve demonstrar compreensão profunda sobre as falhas do sistema:
1. Erros com alta confiança: O modelo tinha certeza da predição mas errou redondamente?
   Isso indica má calibração ou atalhos (shortcuts) nas características.
2. Confusão anatômica: Por que o modelo confundiu Estroma com Músculo ou Tumor com Complexo?
   (Ex: celularidade similar, densidade de matriz extracelular parecida).
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_errors(
    y_true: Union[np.ndarray, List[str]],
    y_pred: Union[np.ndarray, List[str]],
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Identifica todos os índices das amostras classificadas incorretamente.

    Returns:
        Dicionário com estatísticas de erros, índices e lista dos piores erros
        (ordenados por alta confiança na predição errada).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    error_mask = (y_true_arr != y_pred_arr)
    error_indices = np.where(error_mask)[0]

    result: Dict[str, Any] = {
        "total_amostras": len(y_true_arr),
        "total_erros": int(error_mask.sum()),
        "taxa_erro": round(float(error_mask.mean()), 4),
        "indices_erros": error_indices.tolist(),
        "piores_erros": []
    }

    if y_proba is not None and len(error_indices) > 0:
        confidences = y_proba.max(axis=1)
        error_confs = confidences[error_mask]

        result["confianca_media_erros"] = round(float(error_confs.mean()), 4)
        result["erros_alta_confianca_acima_80pct"] = int((error_confs >= 0.80).sum())
        result["erros_alta_confianca_acima_90pct"] = int((error_confs >= 0.90).sum())

        # Ordena erros pelos de maior confiança (mais graves clinicamente)
        order = np.argsort(-error_confs)
        worst_sorted = error_indices[order]
        result["piores_erros"] = [
            {
                "indice": int(idx),
                "rotulo_real": str(y_true_arr[idx]),
                "rotulo_predito": str(y_pred_arr[idx]),
                "confianca": round(float(confidences[idx]), 4)
            }
            for idx in worst_sorted[:20]
        ]

    return result


def plot_error_examples(
    images: List[np.ndarray],
    y_true: List[str],
    y_pred: List[str],
    confidences: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    max_examples: int = 16
) -> None:
    """
    Renderiza um grid ilustrativo com imagens de erro, contrastando o ground truth com a predição.
    """
    n = min(len(images), max_examples)
    if n == 0:
        return

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1 or cols == 1:
        axes_list = list(axes)
    else:
        axes_list = axes.flatten().tolist()

    for i in range(n):
        ax = axes_list[i]
        ax.imshow(images[i])

        title_text = f"Real: {y_true[i]}\nPred: {y_pred[i]}"
        if confidences is not None and i < len(confidences):
            title_text += f" ({confidences[i]:.1%})"

        ax.set_title(title_text, fontsize=9, color="darkred", weight="bold")
        ax.axis("off")

    # Desativa eixos vazios no final do grid
    for j in range(n, len(axes_list)):
        axes_list[j].axis("off")

    plt.suptitle("Galeria de Amostras com Falha de Classificação", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")

    plt.close(fig)


"""
metrics.py — Avaliação quantitativa de desempenho requerida pelo edital do PathoShift.

MÉTRICAS EXIGIDAS:
1. MÉTRICAS POR CLASSE (Per-class Precision, Recall e F1-score):
   - Precision = TP / (TP + FP) — Das amostras preditas como classe C, quantas de fato pertenciam a C?
   - Recall / Sensibilidade = TP / (TP + FN) — De todas as amostras reais da classe C, quantas foram encontradas?
   - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

2. MÉTRICAS MACRO (Macro-averaged):
   - Calcula a métrica para cada classe individualmente e depois tira a média não ponderada:
     Macro F1 = (1 / K) * sum(F1_k)
   - Por que usar Macro e não só Acurácia? Acurácia é enganosa em classes desbalanceadas.
     O Macro F1 penaliza modelos que ignoram classes minoritárias ou mais difíceis.

3. MATRIZ DE CONFUSÃO:
   - Permite visualizar quais classes histológicas causam sobreposição de características.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import numpy as np

# Configura backend do matplotlib para gerar gráficos em lote sem janela gráfica interativa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def compute_metrics(
    y_true: Union[np.ndarray, List[str]],
    y_pred: Union[np.ndarray, List[str]],
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calcula todas as métricas oficiais para a etapa de avaliação.

    Args:
        y_true: Rótulos reais das amostras.
        y_pred: Rótulos previstos pelo classificador.
        y_proba: Matriz de probabilidades de saída (N, num_classes), opcional.
        class_names: Lista ordenada de nomes das classes.

    Returns:
        Dicionário completo contendo acurácia, macro, weighted, métricas por classe e matriz de confusão.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if class_names is None:
        class_names = sorted(list(set(y_true_arr).union(set(y_pred_arr))))

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "f1_macro": round(float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)), 4),
    }

    # Relatório detalhado por classe
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=class_names,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for c in class_names:
        if c in report:
            per_class_metrics[c] = {
                "precision": round(float(report[c]["precision"]), 4),
                "recall": round(float(report[c]["recall"]), 4),
                "f1_score": round(float(report[c]["f1-score"]), 4),
                "support": int(report[c]["support"])
            }
    metrics["per_class"] = per_class_metrics

    # Matriz de Confusão em formato de lista aninhada
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=class_names)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classes"] = class_names

    # ROC-AUC Multiclasse (One-vs-Rest) se houver probabilidades disponíveis
    if y_proba is not None:
        try:
            auc = roc_auc_score(
                y_true_arr,
                y_proba,
                labels=class_names,
                multi_class="ovr",
                average="macro"
            )
            metrics["roc_auc_macro"] = round(float(auc), 4)
        except Exception:
            # Em cenários onde alguma classe não tem amostras ou suporte singular
            pass

    return metrics


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List[str]],
    y_pred: Union[np.ndarray, List[str]],
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True
) -> None:
    """
    Plota e salva uma matriz de confusão elegante com mapa de calor.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        class_names: Lista ordenada de classes.
        save_path: Arquivo PNG para gravação em disco.
        normalize: Se True, normaliza as linhas para exibir proporções (0.00 a 1.00).
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Evita divisão por zero para linhas zeradas
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_data = cm.astype(float) / row_sums
        fmt = ".2f"
        title = "Matriz de Confusão Normalizada (Taxa de Acerto/Erro)"
    else:
        cm_data = cm
        fmt = "d"
        title = "Matriz de Confusão (Contagem Absoluta)"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Rótulo Previsto pelo Modelo", fontsize=11, labelpad=8)
    ax.set_ylabel("Rótulo Verdadeiro (Ground Truth)", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")

    plt.close(fig)


def save_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Exporta o relatório consolidado de métricas em formato JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


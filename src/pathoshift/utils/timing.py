"""
timing.py — Medição de custo de execução por etapa do pipeline.

POR QUE MEDIR TEMPO DE EXECUÇÃO?
O enunciado exige explicitamente o registro do custo de execução.
Isso serve para:
1. Avaliar a viabilidade de aplicar o pipeline em cenários clínicos ou de triagem em larga escala.
2. Identificar gargalos de processamento de imagem (ex: I/O de disco vs. extração LBP vs. treino SVM).
"""

import time
from contextlib import contextmanager
from typing import Dict, Generator, Any


class ExecutionTimer:
    """Cronômetro modular com suporte a context manager para medir blocos de código."""

    def __init__(self) -> None:
        self.records: Dict[str, float] = {}

    @contextmanager
    def measure(self, label: str) -> Generator[None, None, None]:
        """
        Mede o tempo transcorrido durante o bloco 'with'.
        
        Exemplo:
            timer = ExecutionTimer()
            with timer.measure("extracao_features"):
                # codigo
            print(timer.summary())
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            # Acumula se o mesmo label for chamado repetidamente
            self.records[label] = self.records.get(label, 0.0) + elapsed

    def summary(self) -> Dict[str, Any]:
        """Retorna resumo em dicionário com segundos brutos e formatados."""
        total = sum(self.records.values())
        return {
            "etapas_segundos": {k: round(v, 4) for k, v in self.records.items()},
            "etapas_formatado": {k: f"{v:.2f}s" for k, v in self.records.items()},
            "total_segundos": round(total, 4),
            "total_formatado": f"{total:.2f}s"
        }


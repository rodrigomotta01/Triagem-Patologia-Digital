"""
classical.py — Modelo Baseline Clássico com SVM (Support Vector Machine).

POR QUE O SVM É UM EXCELENTE BASELINE EM PROCESSAMENTO DE IMAGENS?
1. Efetivo em espaços de alta dimensão (129 features para 5000 amostras).
2. O kernel RBF (Radial Basis Function) projeta as amostras em uma dimensão superior
   onde hiperplanos não-lineares conseguem separar tecidos com misturas celulares.
3. class_weight='balanced' ajusta automaticamente as margens inversamente proporcionais
   às frequências das classes, prevenindo viés em cenários com desbalanceamento.
4. StandardScaler é obrigatório antes do SVM, pois o algoritmo calcula distâncias euclidianas:
   features com escalas absolutas maiores (ex: médias de brilho) dominariam injustamente
   features de textura que variam entre 0 e 1.
"""

import warnings
import logging
from typing import Optional, Union, List
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suprime avisos de depreciação do scikit-learn para manter log limpo
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class SVMBaseline:
    """Pipeline composto por padronização estatística (z-score) e classificador SVM."""

    def __init__(
        self,
        C: float = 10.0,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        seed: int = 42
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.seed = seed

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                probability=True,        # Permite inferir confianças e probabilidades calibradas
                class_weight="balanced", # Compensação automática de suporte
                random_state=self.seed,
                verbose=False
            ))
        ])
        self.is_fitted = False

    @property
    def classes_(self) -> np.ndarray:
        """Retorna os nomes das classes aprendidas pelo estimador."""
        if not self.is_fitted:
            raise RuntimeError("O modelo ainda não foi treinado.")
        return self.pipeline.named_steps["svm"].classes_

    def fit(self, X: np.ndarray, y: Union[np.ndarray, List[str]]) -> "SVMBaseline":
        """
        Treina o classificador com matriz de características X e vetor de rótulos y.

        Args:
            X: Matriz 2D de formato (N, num_features).
            y: Vetor 1D com rótulos de classe (N,).
        """
        logger.info(f"Treinando SVM com {X.shape[0]} amostras e {X.shape[1]} características.")
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz a classe majoritária para cada amostra."""
        if not self.is_fitted:
            raise RuntimeError("O modelo precisa ser treinado antes de realizar predições.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna as probabilidades estimadas para cada classe.
        Útil para medir a calibração de incerteza e identificar predições ambíguas.
        """
        if not self.is_fitted:
            raise RuntimeError("O modelo precisa ser treinado antes de calcular probabilidades.")
        return self.pipeline.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        """Serializa o pipeline completo treinado em arquivo binário otimizado."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out_path)
        logger.info(f"Modelo SVM serializado com sucesso em {out_path}")

    def load(self, path: Union[str, Path]) -> "SVMBaseline":
        """Carrega os pesos e o padronizador do modelo salvo."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {in_path}")
        self.pipeline = joblib.load(in_path)
        self.is_fitted = True
        logger.info(f"Modelo SVM carregado de {in_path}")
        return self


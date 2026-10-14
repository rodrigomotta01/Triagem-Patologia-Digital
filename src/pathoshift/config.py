"""
config.py — Configuração centralizada do pipeline PathoShift.

POR QUE USAR UM ARQUIVO DE CONFIGURAÇÃO?
- Reprodutibilidade: toda configuração fica registrada em formato YAML.
- Facilidade: permite alterar hiperparâmetros sem modificar o código-fonte.
- Rastreabilidade: cada execução salva a configuração utilizada junto aos resultados.

Usamos dataclasses do Python para obter tipagem estática e valores padrão claros.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    """Configurações relacionadas aos dados e partições."""
    root_dir: str = "./data/kather_texture_2016_image_tiles_5000"
    image_size: int = 150              # Tamanho original das imagens Kather (150x150)
    val_fraction: float = 0.15         # 15% para validação
    test_fraction: float = 0.15        # 15% para teste final


@dataclass
class PreprocessConfig:
    """Configurações de pré-processamento e controle de qualidade."""
    color_norm_method: str = "reinhard" # 'reinhard' ou 'none'
    reference_image: str = ""          # Imagem de referência (se vazia, usa a primeira amostra representativa)
    background_threshold: float = 0.85 # Limiar de proporção de pixels quase brancos para rejeitar fundo
    blur_threshold: float = 100.0      # Variância do Laplaciano mínima para considerar em foco


@dataclass
class FeatureConfig:
    """Configurações dos extratores de features manuais."""
    hsv_bins: int = 32                 # Bins por canal no histograma HSV (32 * 3 = 96 features)
    lbp_radius: int = 3               # Raio do operador LBP (Local Binary Pattern)
    lbp_n_points: int = 24            # Número de pontos vizinhos no LBP
    lbp_bins: int = 26                # Bins no histograma LBP (uniform patterns)


@dataclass
class ModelConfig:
    """Configurações do classificador SVM baseline."""
    svm_C: float = 10.0               # Parâmetro de penalização C da regularização
    svm_kernel: str = "rbf"           # Tipo de kernel ('rbf', 'linear', 'poly')
    svm_gamma: str = "scale"          # Coeficiente do kernel RBF


@dataclass
class ExperimentConfig:
    """Configuração global de um experimento no pipeline."""
    seed: int = 42
    experiment_name: str = "baseline_svm_v1"
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = "./outputs"

    def save(self, path: str) -> None:
        """Salva a configuração atual em um arquivo YAML."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Carrega a configuração a partir de um arquivo YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        data_cfg = DataConfig(**data.get("data", {}))
        preprocess_cfg = PreprocessConfig(**data.get("preprocess", {}))
        features_cfg = FeatureConfig(**data.get("features", {}))
        model_cfg = ModelConfig(**data.get("model", {}))

        return cls(
            seed=data.get("seed", 42),
            experiment_name=data.get("experiment_name", "baseline_svm_v1"),
            data=data_cfg,
            preprocess=preprocess_cfg,
            features=features_cfg,
            model=model_cfg,
            output_dir=data.get("output_dir", "./outputs")
        )


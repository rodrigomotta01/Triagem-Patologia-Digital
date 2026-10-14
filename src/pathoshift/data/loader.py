"""
loader.py — Leitura e validação do dataset Kather (ou qualquer dataset de histopatologia em formato ImageFolder).

ESTRUTURA DO DATASET (ImageFolder):
    data_dir/
    ├── 01_TUMOR/
    │   ├── 10000.tif
    │   └── ...
    ├── 02_STROMA/
    └── ... (8 classes)

POR QUE A VALIDAÇÃO É IMPRESCINDÍVEL?
1. Detectar arquivos corrompidos ou truncados antes de iniciar o treino longo.
2. Conferir consistência de dimensões (ex: se todos os tiles são 150x150).
3. Obter contagens exatas por classe para identificar desbalanceamento.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def read_image_rgb(path: str | Path) -> Optional[np.ndarray]:
    """
    Lê uma imagem e retorna como array NumPy no formato RGB (uint8).
    Utiliza PIL para compatibilidade robusta com caminhos no Windows (incluindo acentos e espaços).
    """
    try:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            return np.array(img_rgb, dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Falha ao ler imagem {path}: {e}")
        return None


def validate_dataset(root_dir: str | Path) -> Dict[str, Any]:
    """
    Valida a integridade do diretório do dataset.

    Returns:
        Dicionário com estatísticas de classes, formatos, dimensões e eventuais arquivos corrompidos.
    """
    root = Path(root_dir)
    stats: Dict[str, Any] = {
        "root_dir": str(root.resolve()),
        "total_images": 0,
        "classes": {},
        "dimensions": set(),
        "formats": set(),
        "corrupted": [],
    }

    if not root.exists() or not root.is_dir():
        logger.error(f"Diretório raiz não encontrado: {root}")
        return stats

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        count = 0

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify() # Verifica integridade estrutural
                # Reabre para inspecionar dimensões e formato
                with Image.open(img_path) as img:
                    stats["dimensions"].add(img.size)
                    fmt = img.format if img.format else img_path.suffix.upper().replace(".", "")
                    stats["formats"].add(fmt)
                    count += 1
            except Exception as e:
                stats["corrupted"].append(str(img_path))
                logger.warning(f"Arquivo corrompido: {img_path} ({e})")

        stats["classes"][class_name] = count
        stats["total_images"] += count

    # Converte sets para listas para permitir serialização JSON
    stats["dimensions"] = [list(dim) for dim in stats["dimensions"]]
    stats["formats"] = sorted(list(stats["formats"]))

    return stats


def load_images_and_labels(
    root_dir: str | Path,
    max_per_class: Optional[int] = None
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Carrega as imagens RGB para a memória junto aos seus rótulos e caminhos originais.

    Args:
        root_dir: Caminho para a pasta raiz das classes.
        max_per_class: Opcional, para testes rápidos com um subconjunto de imagens.

    Returns:
        images: Lista de arrays NumPy RGB (H, W, 3).
        labels: Lista de strings com os nomes das classes.
        paths: Lista de strings com os caminhos dos arquivos.
    """
    root = Path(root_dir)
    images: List[np.ndarray] = []
    labels: List[str] = []
    paths: List[str] = []

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Diretório do dataset não existe: {root}")

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        count = 0

        for img_path in sorted(class_dir.iterdir()):
            if max_per_class is not None and count >= max_per_class:
                break

            if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            img_rgb = read_image_rgb(img_path)
            if img_rgb is not None:
                images.append(img_rgb)
                labels.append(class_name)
                paths.append(str(img_path))
                count += 1

    logger.info(f"Carregadas {len(images)} imagens pertencentes a {len(set(labels))} classes distintas.")
    return images, labels, paths


def split_dataset(
    images: List[np.ndarray],
    labels: List[str],
    paths: List[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Realiza particionamento estratificado em Treino, Validação e Teste.

    Estratificação garante que todas as 8 classes mantenham proporções idênticas
    nas 3 partições, impedindo desbalanço espúrio no conjunto de teste.
    """
    if len(images) == 0:
        raise ValueError("Lista de imagens fornecida está vazia.")

    # 1. Separar o conjunto de Teste final
    X_train_val, X_test, y_train_val, y_test, p_train_val, p_test = train_test_split(
        images, labels, paths,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels
    )

    # 2. Separar Validação a partir do restante
    val_relative = val_fraction / (1.0 - test_fraction)
    X_train, X_val, y_train, y_val, p_train, p_val = train_test_split(
        X_train_val, y_train_val, p_train_val,
        test_size=val_relative,
        random_state=seed,
        stratify=y_train_val
    )

    return {
        "train": {"images": X_train, "labels": y_train, "paths": p_train},
        "val":   {"images": X_val,   "labels": y_val,   "paths": p_val},
        "test":  {"images": X_test,  "labels": y_test,  "paths": p_test},
    }


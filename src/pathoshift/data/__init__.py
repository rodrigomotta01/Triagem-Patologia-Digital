"""
Módulo de manipulação de dados: validação de integridade, carregamento e manifesto JSONL.
"""

from .loader import validate_dataset, load_images_and_labels, split_dataset
from .manifest import generate_manifest

__all__ = [
    "validate_dataset",
    "load_images_and_labels",
    "split_dataset",
    "generate_manifest"
]


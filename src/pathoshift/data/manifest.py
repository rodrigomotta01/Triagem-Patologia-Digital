"""
manifest.py — Geração de manifesto reprodutível em formato JSONL (JSON Lines).

O QUE É UM MANIFESTO REPRODUTÍVEL?
É um inventário estruturado onde cada linha representa exatamente uma imagem do conjunto
com seus metadados (caminho relativo, rótulo, dimensões e hash SHA-256 para integridade).

POR QUE FORMATO JSONL?
- Cada linha é um objeto JSON independente e autossuficiente.
- Permite processamento em streaming (sem carregar arquivo inteiro na RAM).
- Facilita auditorias, versionamento e verificação de duplicatas por hash criptográfico.
"""

from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """Calcula o resumo criptográfico SHA-256 de um arquivo em disco."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifest(
    root_dir: str | Path,
    output_path: str | Path,
    split_name: Optional[str] = None
) -> int:
    """
    Varre o diretório do dataset e grava o arquivo de manifesto JSONL.

    Args:
        root_dir: Diretório raiz do dataset.
        output_path: Caminho completo para salvar o arquivo .jsonl.
        split_name: Opcional, nome da partição ('train', 'val', 'test', etc.).

    Returns:
        Número total de registros gravados.
    """
    root = Path(root_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    with Image.open(img_path) as img:
                        width, height = img.size

                    file_hash = compute_file_hash(img_path)
                    rel_path = str(img_path.relative_to(root)).replace("\\", "/")

                    record: Dict[str, Any] = {
                        "image_path": rel_path,
                        "label": class_name,
                        "width": width,
                        "height": height,
                        "file_hash_sha256": file_hash,
                        "dataset_root": str(root.resolve()).replace("\\", "/"),
                        "created_at": datetime.now().isoformat(),
                    }

                    if split_name:
                        record["split"] = split_name

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    logger.warning(f"Erro ao incluir {img_path} no manifesto: {e}")

    logger.info(f"Manifesto gerado com sucesso: {count} registros em {out_file}")
    return count


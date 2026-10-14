"""
create_sample_dataset.py — Gera um subconjunto de amostras sintéticas no formato Kather
para demonstração, testes locais imediatos e validação da CLI.
"""

from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = [
    "01_TUMOR",
    "02_STROMA",
    "03_COMPLEX",
    "04_LYMPHO",
    "05_DEBRIS",
    "06_MUCOSA",
    "07_ADIPOSE",
    "08_EMPTY"
]


def generate_sample_tile(class_name: str, seed: int) -> np.ndarray:
    """Gera uma imagem sintética de 150x150 com propriedades visuais da classe."""
    rng = np.random.default_rng(seed)
    H, W = 150, 150

    if class_name == "08_EMPTY":
        # Vidro da lâmina: quase totalmente branco (>230) com leve ruído de iluminação
        base = rng.integers(235, 255, size=(H, W, 3), dtype=np.uint8)
        return base

    elif class_name == "07_ADIPOSE":
        # Tecido adiposo: grandes áreas brancas circulares (gordura) delimitadas por septos rosados
        img = rng.integers(230, 250, size=(H, W, 3), dtype=np.uint8)
        # Traça linhas simulando membranas
        for _ in range(8):
            y = rng.integers(0, H)
            img[max(0, y-1):min(H, y+2), :, :] = [180, 100, 160] # rosa/violeta
        return img

    elif class_name == "04_LYMPHO":
        # Linfócitos: alta densidade de núcleos celulares esféricos pequenos e intensamente roxos
        img = rng.integers(150, 180, size=(H, W, 3), dtype=np.uint8) # fundo rosado
        # núcleos escuros
        num_nuclei = rng.integers(80, 120)
        for _ in range(num_nuclei):
            cy, cx = rng.integers(5, H-5), rng.integers(5, W-5)
            img[cy-2:cy+3, cx-2:cx+3] = [80, 30, 110] # roxo escuro basofílico
        return img

    elif class_name == "01_TUMOR":
        # Epitélio tumoral: núcleos aglomerados, citoplasma denso, tom roxo/azul predominante
        base = np.zeros((H, W, 3), dtype=np.uint8)
        base[:, :, 0] = rng.integers(120, 150, size=(H, W)) # R
        base[:, :, 1] = rng.integers(40, 80, size=(H, W))   # G
        base[:, :, 2] = rng.integers(140, 180, size=(H, W)) # B
        # Textura irregular de células epiteliais
        for _ in range(40):
            y, x = rng.integers(0, H-10), rng.integers(0, W-10)
            base[y:y+8, x:x+8] = [100, 30, 130]
        return base

    elif class_name == "02_STROMA":
        # Estroma: colágeno fibroso rosado (eosina alta)
        base = np.zeros((H, W, 3), dtype=np.uint8)
        base[:, :, 0] = rng.integers(190, 220, size=(H, W)) # R alto
        base[:, :, 1] = rng.integers(90, 120, size=(H, W))  # G médio
        base[:, :, 2] = rng.integers(150, 180, size=(H, W)) # B médio
        return base

    elif class_name == "06_MUCOSA":
        # Glândulas organizadas
        base = np.full((H, W, 3), [170, 80, 160], dtype=np.uint8)
        # Estrutura glandular elíptica clara no centro
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - 75)**2 / 30**2 + (xx - 75)**2 / 40**2) < 1
        base[mask] = [220, 200, 225]
        return base

    else:
        # COMPLEX / DEBRIS: Mistura heterogênea
        base = rng.integers(110, 190, size=(H, W, 3), dtype=np.uint8)
        return base


def populate_sample_dataset(
    target_dir: str = "./data/kather_texture_2016_image_tiles_5000",
    images_per_class: int = 15
) -> None:
    root = Path(target_dir)
    print(f"[*] Gerando dataset de demonstração em: {root.resolve()}")

    total = 0
    for c_idx, class_name in enumerate(CLASSES):
        class_folder = root / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        for i in range(images_per_class):
            seed = c_idx * 1000 + i
            tile = generate_sample_tile(class_name, seed)
            img = Image.fromarray(tile)
            file_path = class_folder / f"sample_{i+1:04d}.tif"
            img.save(file_path)
            total += 1

    print(f"[+] Total de {total} imagens sintéticas geradas com sucesso em {len(CLASSES)} classes!")


if __name__ == "__main__":
    populate_sample_dataset()


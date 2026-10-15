"""
download_dataset.py — Script para download e extração automática do dataset oficial
Kather Colorectal Histology (5.000 tiles, ~258 MB) diretamente do repositório Zenodo.
"""

import sys
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip"
TARGET_DIR = Path("./data")
ZIP_DEST = TARGET_DIR / "Kather_texture_2016_image_tiles_5000.zip"
EXTRACT_FOLDER = TARGET_DIR / "kather_texture_2016_image_tiles_5000"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("DOWNLOAD AUTOMÁTICO DO DATASET KATHER (5.000 TILES - H&E)")
    print(f"Fonte Oficial: {ZENODO_URL}")
    print("=" * 65)

    if EXTRACT_FOLDER.exists() and len(list(EXTRACT_FOLDER.glob("*/*.tif"))) >= 5000:
        print(f"[!] O dataset completo já está presente em: {EXTRACT_FOLDER.resolve()}")
        print("[!] Nenhuma ação necessária.")
        return

    # 1. Download
    if not ZIP_DEST.exists() or ZIP_DEST.stat().st_size < 250_000_000:
        print(f"[*] Baixando arquivo ZIP (~258 MB) para {ZIP_DEST}...")
        try:
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Download") as t:
                urllib.request.urlretrieve(ZENODO_URL, filename=ZIP_DEST, reporthook=t.update_to)
            print("[+] Download concluído com sucesso!")
        except Exception as e:
            print(f"[!] Erro durante o download: {e}")
            print("[!] Você também pode baixar manualmente em:")
            print("    - Zenodo: https://doi.org/10.5281/zenodo.53169")
            print("    - Kaggle: https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist")
            sys.exit(1)
    else:
        print(f"[*] Arquivo ZIP já encontrado em {ZIP_DEST}. Pulando download.")

    # 2. Extração
    print(f"[*] Extraindo arquivos em {TARGET_DIR.resolve()}...")
    try:
        with zipfile.ZipFile(ZIP_DEST, "r") as zip_ref:
            zip_ref.extractall(TARGET_DIR)
        print(f"[+] Extração concluída com sucesso!")

        total_imgs = len(list(EXTRACT_FOLDER.glob("*/*.tif")))
        print(f"[+] Total de imagens prontas para treino: {total_imgs}")
        print("\nPara iniciar o treino completo agora, execute:")
        print("  py -m pathoshift train --config ./configs/default.yaml")

    except Exception as e:
        print(f"[!] Erro ao descompactar arquivo ZIP: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_and_extract()


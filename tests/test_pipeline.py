"""
test_pipeline.py — Testes unitários e de integração de ponta a ponta para o PathoShift.
Gera dados sintéticos no padrão H&E (150x150) e testa todas as etapas do pipeline.
"""

import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Adiciona pasta 'src' ao path de execução
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathoshift.config import ExperimentConfig
from pathoshift.data.loader import validate_dataset, load_images_and_labels, split_dataset
from pathoshift.data.manifest import generate_manifest
from pathoshift.preprocessing.quality import detect_background, assess_focus, quality_report
from pathoshift.preprocessing.color_norm import ReinhardNormalizer
from pathoshift.features.extractor import extract_all_features
from pathoshift.models.classical import SVMBaseline
from pathoshift.evaluation.metrics import compute_metrics, plot_confusion_matrix
from pathoshift.evaluation.error_analysis import find_errors, plot_error_examples


def create_synthetic_dataset(data_dir: Path, num_classes: int = 8, per_class: int = 5) -> None:
    """Gera um mini-dataset sintético simulando coloração H&E e fundo."""
    class_names = [
        "01_TUMOR", "02_STROMA", "03_COMPLEX", "04_LYMPHO",
        "05_DEBRIS", "06_MUCOSA", "07_ADIPOSE", "08_EMPTY"
    ][:num_classes]

    if data_dir.exists():
        shutil.rmtree(data_dir)

    rng = np.random.default_rng(42)

    for c_idx, c_name in enumerate(class_names):
        c_folder = data_dir / c_name
        c_folder.mkdir(parents=True, exist_ok=True)

        for i in range(per_class):
            if c_name == "08_EMPTY":
                # Quase todo branco
                arr = rng.integers(230, 255, size=(150, 150, 3), dtype=np.uint8)
            else:
                # Simula tons de roxo/rosa (H&E): R alto, G médio, B alto
                base_r = int(140 + c_idx * 10)
                base_g = int(60 + c_idx * 5)
                base_b = int(150 + c_idx * 8)
                noise = rng.integers(-30, 30, size=(150, 150, 3))
                base = np.array([base_r, base_g, base_b], dtype=np.int16)
                arr = np.clip(base + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(arr)
            img.save(c_folder / f"tile_{i:04d}.tif")


def run_all_tests() -> None:
    print("\n" + "=" * 60)
    print("INICIANDO SUÍTE DE TESTES DO PIPELINE PATHOSHIFT")
    print("=" * 60)

    test_data_dir = Path("./tests/temp_test_data")
    test_out_dir = Path("./tests/temp_test_output")
    test_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Criação de Dados Sintéticos
        print("[1] Gerando mini-dataset sintético...")
        create_synthetic_dataset(test_data_dir, num_classes=8, per_class=5)
        print("    -> Dataset sintético criado com 40 amostras.")

        # 2. Validação de Dataset
        print("[2] Testando validate_dataset...")
        stats = validate_dataset(test_data_dir)
        assert stats["total_images"] == 40, f"Esperado 40 imagens, obteve {stats['total_images']}"
        assert len(stats["classes"]) == 8, f"Esperado 8 classes, obteve {len(stats['classes'])}"
        print("    -> Sucesso! Validação íntegra.")

        # 3. Geração de Manifesto JSONL
        print("[3] Testando generate_manifest...")
        manifest_path = test_out_dir / "manifest.jsonl"
        count = generate_manifest(test_data_dir, manifest_path)
        assert count == 40
        assert manifest_path.exists()
        print(f"    -> Sucesso! Manifesto gerado com {count} linhas.")

        # 4. Controle de Qualidade
        print("[4] Testando detect_background e assess_focus...")
        white_img = np.full((150, 150, 3), 245, dtype=np.uint8)
        is_bg, ratio = detect_background(white_img)
        assert is_bg and ratio > 0.9
        is_foc, var = assess_focus(white_img)
        assert not is_foc, "Imagem plana sem textura não deve estar em foco."
        print("    -> Sucesso! Controle de qualidade validado.")

        # 5. Normalização de Cor Reinhard
        print("[5] Testando ReinhardNormalizer...")
        ref_img = np.full((150, 150, 3), [180, 50, 150], dtype=np.uint8)
        src_img = np.full((150, 150, 3), [100, 100, 100], dtype=np.uint8)
        normalizer = ReinhardNormalizer().fit(ref_img)
        norm_result = normalizer.transform(src_img)
        assert norm_result.shape == (150, 150, 3)
        assert norm_result.dtype == np.uint8
        print("    -> Sucesso! Normalizador de cor funcional.")

        # 6. Extração de Features
        print("[6] Testando extract_all_features...")
        dummy_img = np.random.randint(0, 255, size=(150, 150, 3), dtype=np.uint8)
        features = extract_all_features(dummy_img)
        assert features.shape == (129,), f"Esperado 129 dimensões, obteve {features.shape}"
        assert not np.isnan(features).any(), "Features contêm valores NaN!"
        print("    -> Sucesso! Vetor de características exato com 129 dimensões.")

        # 7. Carregamento e Split
        print("[7] Testando load_images_and_labels e split_dataset...")
        imgs, lbls, paths = load_images_and_labels(test_data_dir)
        splits = split_dataset(imgs, lbls, paths, val_fraction=0.2, test_fraction=0.2, seed=42)
        assert len(splits["train"]["images"]) > 0
        assert len(splits["val"]["images"]) > 0
        assert len(splits["test"]["images"]) > 0
        print("    -> Sucesso! Particionamento estratificado funcional.")

        # 8. Treinamento e Predição SVM
        print("[8] Testando SVMBaseline fit e predict...")
        X_train = np.array([extract_all_features(img) for img in splits["train"]["images"]])
        y_train = splits["train"]["labels"]
        X_test = np.array([extract_all_features(img) for img in splits["test"]["images"]])
        y_test = splits["test"]["labels"]

        svm = SVMBaseline(C=1.0, seed=42).fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        y_proba = svm.predict_proba(X_test)
        assert len(y_pred) == len(y_test)
        assert y_proba.shape == (len(y_test), len(svm.classes_))

        model_file = test_out_dir / "test_model.joblib"
        svm.save(model_file)
        svm_reloaded = SVMBaseline().load(model_file)
        assert (svm_reloaded.predict(X_test) == y_pred).all()
        print("    -> Sucesso! Treino, inferência e serialização do SVM aprovados.")

        # 9. Cálculo de Métricas e Matriz de Confusão
        print("[9] Testando compute_metrics e plot_confusion_matrix...")
        metrics = compute_metrics(y_test, y_pred, y_proba, class_names=sorted(list(svm.classes_)))
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics

        cm_png = test_out_dir / "test_cm.png"
        plot_confusion_matrix(y_test, y_pred, class_names=metrics["classes"], save_path=cm_png)
        assert cm_png.exists()
        print(f"    -> Sucesso! Métricas computadas (Acurácia: {metrics['accuracy']:.2f}, F1: {metrics['f1_macro']:.2f}).")

        # 10. Análise de Erros
        print("[10] Testando find_errors e plot_error_examples...")
        err_info = find_errors(y_test, y_pred, y_proba)
        assert "total_erros" in err_info
        plot_error_examples(splits["test"]["images"][:4], y_test[:4], y_pred[:4], save_path=test_out_dir / "test_err.png")
        print("    -> Sucesso! Análise de erros funcional.")

        print("\n" + "=" * 60)
        print("TODOS OS TESTES UNITÁRIOS E DE INTEGRAÇÃO PASSARAM COM SUCESSO!")
        print("=" * 60 + "\n")

    finally:
        # Limpeza
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        if test_out_dir.exists():
            shutil.rmtree(test_out_dir)


if __name__ == "__main__":
    run_all_tests()


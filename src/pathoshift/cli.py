"""
cli.py — Interface de Linha de Comando (CLI) oficial do PathoShift Challenge.

COMANDOS SUPORTADOS:
  python -m pathoshift describe
  python -m pathoshift prepare  --input <data_dir>   --output <output_dir>
  python -m pathoshift train    --config <config.yaml>
  python -m pathoshift predict  --input <manifest.jsonl> --output <predictions.jsonl> --model <model.joblib>
  python -m pathoshift evaluate --input <predictions.jsonl> --output <metrics.json>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

from pathoshift.config import ExperimentConfig
from pathoshift.utils.reproducibility import set_seed, get_environment_info
from pathoshift.utils.timing import ExecutionTimer
from pathoshift.data.loader import validate_dataset, load_images_and_labels, split_dataset, read_image_rgb
from pathoshift.data.manifest import generate_manifest
from pathoshift.preprocessing.quality import detect_background, assess_focus
from pathoshift.preprocessing.color_norm import ReinhardNormalizer
from pathoshift.features.extractor import extract_all_features
from pathoshift.models.classical import SVMBaseline
from pathoshift.evaluation.metrics import compute_metrics, plot_confusion_matrix, save_metrics
from pathoshift.evaluation.error_analysis import find_errors, plot_error_examples


def cmd_describe(args: argparse.Namespace) -> None:
    """Imprime especificações técnicas e ambiente do sistema em formato JSON."""
    info = {
        "sistema": "PathoShift Challenge Pipeline",
        "etapa": "Entrega 1 — Pathology Image Pipeline",
        "dataset_alvo": "Kather Colorectal Histology (5.000 tiles, 8 classes)",
        "baseline": "Classificador Clássico: StandardScaler + SVM com Kernel RBF",
        "espaco_caracteristicas": {
            "dimensao_total": 129,
            "histograma_hsv": "32 bins * 3 canais = 96 features",
            "textura_lbp": "Uniform LBP (R=3, P=24) = 26 features",
            "estatisticas_globais": "Media e Desvio RGB (6) + Entropia de Shannon (1) = 7 features"
        },
        "pre_processamento": {
            "normalizacao_cor": "Reinhard L*a*b* Color Transfer",
            "controle_qualidade": "Rejeicao de fundo branco (>85%) e medicao de foco via Laplaciano"
        },
        "ambiente": get_environment_info()
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))


def cmd_prepare(args: argparse.Namespace) -> None:
    """Valida o conjunto de imagens e gera o manifesto oficial JSONL."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Validando imagens em: {input_path}")
    stats = validate_dataset(input_path)
    print(f"[*] Total de imagens encontradas: {stats['total_images']}")
    print(f"[*] Classes detectadas ({len(stats['classes'])}): {list(stats['classes'].keys())}")

    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[+] Estatísticas salvas em: {stats_file}")

    manifest_file = output_dir / "manifest.jsonl"
    total_records = generate_manifest(input_path, manifest_file)
    print(f"[+] Manifesto JSONL gerado com sucesso ({total_records} registros): {manifest_file}")


def cmd_train(args: argparse.Namespace) -> None:
    """Executa o pipeline completo: pré-processamento, extração de features, treino e avaliação."""
    config_path = args.config if args.config else "./configs/default.yaml"
    cfg = ExperimentConfig.load(config_path) if Path(config_path).exists() else ExperimentConfig()

    set_seed(cfg.seed)
    timer = ExecutionTimer()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar Dados
    print(f"[*] 1/6 Carregando dataset a partir de: {cfg.data.root_dir}")
    with timer.measure("1_carregamento_dados"):
        images, labels, paths = load_images_and_labels(cfg.data.root_dir)

    if len(images) == 0:
        print("[!] ERRO: Nenhuma imagem encontrada no diretório especificado.")
        sys.exit(1)

    # 2. Particionamento
    print(f"[*] 2/6 Dividindo dataset em Treino ({1 - cfg.data.val_fraction - cfg.data.test_fraction:.0%}), "
          f"Validação ({cfg.data.val_fraction:.0%}) e Teste ({cfg.data.test_fraction:.0%})")
    with timer.measure("2_split_dataset"):
        splits = split_dataset(
            images, labels, paths,
            val_fraction=cfg.data.val_fraction,
            test_fraction=cfg.data.test_fraction,
            seed=cfg.seed
        )

    # 3. Pré-processamento (Normalização Reinhard)
    normalizer = ReinhardNormalizer()
    if cfg.preprocess.color_norm_method == "reinhard":
        print("[*] 3/6 Ajustando normalizador de cor Reinhard com amostra de referência...")
        with timer.measure("3_ajuste_reinhard"):
            # Usa primeira imagem de treino como padrão de coloração
            ref_img = splits["train"]["images"][0]
            normalizer.fit(ref_img)

    # Função auxiliar de extração
    def process_split_features(split_name: str) -> np.ndarray:
        raw_imgs = splits[split_name]["images"]
        feats_list = []
        for img in tqdm(raw_imgs, desc=f"Features ({split_name})"):
            if cfg.preprocess.color_norm_method == "reinhard":
                img = normalizer.transform(img)
            f = extract_all_features(
                img,
                hsv_bins=cfg.features.hsv_bins,
                lbp_radius=cfg.features.lbp_radius,
                lbp_n_points=cfg.features.lbp_n_points,
                lbp_bins=cfg.features.lbp_bins
            )
            feats_list.append(f)
        return np.array(feats_list, dtype=np.float32)

    # 4. Extração de Features
    print("[*] 4/6 Extraindo características (HSV + LBP + Estatísticas)...")
    with timer.measure("4_extracao_features_treino"):
        X_train = process_split_features("train")
        y_train = splits["train"]["labels"]

    with timer.measure("4_extracao_features_val"):
        X_val = process_split_features("val")
        y_val = splits["val"]["labels"]

    with timer.measure("4_extracao_features_teste"):
        X_test = process_split_features("test")
        y_test = splits["test"]["labels"]

    # 5. Treinamento SVM
    print(f"[*] 5/6 Treinando SVM (Kernel: {cfg.model.svm_kernel}, C: {cfg.model.svm_C})...")
    model = SVMBaseline(
        C=cfg.model.svm_C,
        kernel=cfg.model.svm_kernel,
        gamma=cfg.model.svm_gamma,
        seed=cfg.seed
    )
    with timer.measure("5_treinamento_svm"):
        model.fit(X_train, y_train)

    model_path = out_dir / "svm_baseline.joblib"
    model.save(model_path)

    # 6. Avaliação no conjunto de Teste
    print("[*] 6/6 Avaliando predições no conjunto de Teste...")
    with timer.measure("6_inferencia_teste"):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    class_names = sorted(list(model.classes_))
    metrics = compute_metrics(y_test, y_pred, y_proba, class_names=class_names)
    metrics["custo_execucao"] = timer.summary()

    metrics_file = out_dir / "metrics.json"
    save_metrics(metrics, metrics_file)

    cm_plot_file = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, class_names=class_names, save_path=cm_plot_file, normalize=True)

    # Análise de Erros
    errors_info = find_errors(y_test, y_pred, y_proba)
    with open(out_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(errors_info, f, indent=2, ensure_ascii=False)

    # Plotar exemplos dos erros
    error_indices = errors_info["indices_erros"]
    if error_indices:
        err_imgs = [splits["test"]["images"][i] for i in error_indices[:16]]
        err_true = [splits["test"]["labels"][i] for i in error_indices[:16]]
        err_pred = [y_pred[i] for i in error_indices[:16]]
        err_conf = [float(y_proba[i].max()) for i in error_indices[:16]]
        plot_error_examples(err_imgs, err_true, err_pred, err_conf, save_path=out_dir / "error_examples.png")

    print(f"\n[+] Treino e Avaliação concluídos!")
    print(f"[+] Acurácia no Teste: {metrics['accuracy']:.2%}")
    print(f"[+] F1-Score Macro:   {metrics['f1_macro']:.2%}")
    print(f"[+] Métricas salvas em: {metrics_file}")
    print(f"[+] Matriz de confusão em: {cm_plot_file}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Gera predições a partir de um manifesto JSONL e salva predictions.jsonl."""
    manifest_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = args.model if args.model else "./outputs/svm_baseline.joblib"
    model = SVMBaseline().load(model_path)

    # Carrega registros do manifesto
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Assume que paths relativos são relativos ao diretório do manifesto ou cwd
    parent_dir = manifest_path.parent

    predictions = []
    print(f"[*] Gerando inferências para {len(records)} imagens...")

    for rec in tqdm(records, desc="Inferencia"):
        img_rel = rec.get("image_path", "")
        dataset_root = rec.get("dataset_root")

        # Tenta resolver o caminho real da imagem
        candidates = []
        if getattr(args, "data_dir", None):
            candidates.append(Path(args.data_dir) / img_rel)
        if dataset_root:
            candidates.append(Path(dataset_root) / img_rel)
        candidates.append(parent_dir / img_rel)
        candidates.append(Path(img_rel))
        candidates.append(Path("./data/kather_texture_2016_image_tiles_5000") / img_rel)

        img_file = None
        for cand in candidates:
            if cand.exists():
                img_file = cand
                break

        if img_file is None:
            continue

        img_rgb = read_image_rgb(img_file)
        if img_rgb is None:
            continue

        feat = extract_all_features(img_rgb).reshape(1, -1)
        pred_label = str(model.predict(feat)[0])
        proba = model.predict_proba(feat)[0]
        confidence = float(proba.max())

        out_rec = {
            "image_path": img_rel,
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "ground_truth": rec.get("label", None)
        }
        predictions.append(out_rec)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[+] Predições salvas em: {output_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Lê predictions.jsonl ou manifesto anotado e calcula métricas completas."""
    pred_path = Path(args.input)
    output_path = Path(args.output)

    records = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    y_true = []
    y_pred = []

    for r in records:
        gt = r.get("ground_truth") or r.get("label")
        pred = r.get("prediction")
        if gt and pred:
            y_true.append(gt)
            y_pred.append(pred)

    if not y_true:
        print("[!] Não foram encontrados rótulos verdadeiros e predições pareadas no arquivo fornecido.")
        sys.exit(1)

    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics, output_path)

    cm_path = output_path.parent / "evaluation_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names=metrics["classes"], save_path=cm_path)

    print(f"[+] Avaliação calculada com sucesso!")
    print(f"[+] Acurácia: {metrics['accuracy']:.2%}")
    print(f"[+] F1 Macro: {metrics['f1_macro']:.2%}")
    print(f"[+] Arquivo salvo: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pathoshift",
        description="PathoShift Challenge: Pipeline de Triagem Robusta em Patologia Digital"
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando a ser executado")

    # describe
    subparsers.add_parser("describe", help="Exibe resumo da arquitetura do sistema e ambiente")

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Valida dataset e gera manifesto JSONL")
    p_prep.add_argument("--input", required=True, help="Diretório raiz das classes de imagens")
    p_prep.add_argument("--output", required=True, help="Diretório de saída para manifesto e estatísticas")

    # train
    p_train = subparsers.add_parser("train", help="Executa o pipeline completo de treinamento do SVM")
    p_train.add_argument("--config", default="./configs/default.yaml", help="Caminho do arquivo YAML de configuração")

    # predict
    p_pred = subparsers.add_parser("predict", help="Realiza predições para imagens de um manifesto JSONL")
    p_pred.add_argument("--input", required=True, help="Arquivo manifest.jsonl de entrada")
    p_pred.add_argument("--output", required=True, help="Arquivo predictions.jsonl de saída")
    p_pred.add_argument("--model", default="./outputs/svm_baseline.joblib", help="Arquivo do modelo SVM treinado")
    p_pred.add_argument("--data-dir", default=None, help="Diretório raiz alternativo das imagens (se necessário)")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Gera métricas a partir de predictions.jsonl")
    p_eval.add_argument("--input", required=True, help="Arquivo predictions.jsonl")
    p_eval.add_argument("--output", required=True, help="Arquivo metrics.json de saída")

    args = parser.parse_args()

    commands = {
        "describe": cmd_describe,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


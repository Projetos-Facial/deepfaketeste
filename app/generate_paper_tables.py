"""
Gera todos os CSVs e tabelas consolidadas para o paper.

Preenche os 3 itens faltantes:
  1. personal_swap_attack_results.csv  — scores dos swaps do dataset pessoal
  2. baseline_metrics.csv              — FAR/FRR do baseline em formato tabular
  3. consolidated_results.csv          — tabela antes vs depois do ataque

Execução:
  python app/generate_paper_tables.py
"""

import os
import sys
import csv
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_lfw import (
    ADAFACE_CKPT, YOLO_FACE_PATH, FACE_SIZE, DEVICE,
    load_adaface, detect_and_crop_face, preprocess_face, get_embedding,
)
from ultralytics import YOLO

RESULTS_DIR = "app/results"
PERSONAL_ORIGINALS = "data/dataset_pessoal/originais"
PERSONAL_SWAPS = "data/dataset_pessoal/swaps"
THRESHOLDS = [0.45, 0.60]


def generate_personal_attack_csv(yolo, adaface):
    """
    Item 1: Avalia cada swap do dataset pessoal contra a foto original
    da pessoa-alvo e salva os resultados em CSV.
    """
    print("\n" + "=" * 70)
    print(" 1/3  CSV DO ATAQUE NO DATASET PESSOAL ".center(70, "═"))
    print("=" * 70)

    if not os.path.isdir(PERSONAL_SWAPS):
        print("⚠ Pasta de swaps pessoais não encontrada. Pulando...")
        return

    swap_files = sorted([
        f for f in os.listdir(PERSONAL_SWAPS)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not swap_files:
        print("⚠ Nenhum swap encontrado em", PERSONAL_SWAPS)
        return

    results = []

    for sf in swap_files:
        swap_path = os.path.join(PERSONAL_SWAPS, sf)

        # Extrair nome da pessoa-alvo (source) do nome do arquivo
        # Formato: swap_<source>_in_<target>.jpg
        parts = sf.replace("swap_", "").replace(".jpg", "").split("_in_")
        if len(parts) != 2:
            print(f"  ⚠ Nome inesperado: {sf}, pulando...")
            continue

        source_name = parts[0]  # pessoa cujo rosto foi colado
        target_name = parts[1]  # pessoa cujo corpo foi usado

        # Encontrar a foto original da pessoa-alvo (source)
        original_candidates = [
            f for f in os.listdir(PERSONAL_ORIGINALS)
            if f.lower().startswith(source_name.lower()) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not original_candidates:
            print(f"  ⚠ Foto original de '{source_name}' não encontrada, pulando...")
            continue

        original_path = os.path.join(PERSONAL_ORIGINALS, original_candidates[0])

        # Carregar imagens
        img_swap = cv2.imread(swap_path)
        img_orig = cv2.imread(original_path)

        if img_swap is None or img_orig is None:
            print(f"  ⚠ Erro ao carregar imagens para {sf}")
            continue

        # Extrair embeddings
        face_swap = detect_and_crop_face(yolo, img_swap, FACE_SIZE)
        face_orig = detect_and_crop_face(yolo, img_orig, FACE_SIZE)

        emb_swap = get_embedding(adaface, preprocess_face(face_swap), DEVICE)
        emb_orig = get_embedding(adaface, preprocess_face(face_orig), DEVICE)

        sim = float(np.dot(emb_swap, emb_orig) / (
            np.linalg.norm(emb_swap) * np.linalg.norm(emb_orig) + 1e-8
        ))

        results.append({
            "swap_file": sf,
            "source_person": source_name,
            "target_person": target_name,
            "original_file": original_candidates[0],
            "similarity": sim,
            "fooled_045": int(sim >= 0.45),
            "fooled_060": int(sim >= 0.60),
        })

        status = "🚨 ENGANOU" if sim >= 0.45 else "🛡️ BARRADO"
        print(f"  {sf:55s}  sim={sim:.4f}  {status}")

    # Salvar CSV
    csv_path = os.path.join(RESULTS_DIR, "personal_swap_attack_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  ✅ Salvo em: {csv_path}")

    # Mostrar resumo
    sims = [r["similarity"] for r in results]
    n = len(sims)
    for tau in THRESHOLDS:
        n_fooled = sum(1 for s in sims if s >= tau)
        print(f"  ASR (τ={tau}): {n_fooled}/{n} = {n_fooled/n*100:.1f}%")

    return results


def generate_baseline_metrics_csv():
    """
    Item 2: Extrai FAR/FRR do baseline em formato CSV limpo.
    """
    print("\n" + "=" * 70)
    print(" 2/3  CSV DE MÉTRICAS DO BASELINE ".center(70, "═"))
    print("=" * 70)

    # Ler os scores do baseline
    scores_path = os.path.join(RESULTS_DIR, "lfw_scores.csv")
    if not os.path.exists(scores_path):
        print("⚠ lfw_scores.csv não encontrado. Rode baseline_lfw.py primeiro.")
        return

    df = pd.read_csv(scores_path)
    scores = df["score"].values
    labels = df["label"].values

    genuine = scores[labels == 1]
    impostor = scores[labels == 0]

    # Calcular métricas
    from sklearn.metrics import roc_curve, auc, accuracy_score

    fpr, tpr, thresholds_roc = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]

    # Melhor threshold
    best_acc, best_t = 0, 0
    for t in np.arange(0.0, 1.0, 0.001):
        preds = (scores >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t

    # FAR/FRR por threshold
    rows = []
    for tau in THRESHOLDS:
        # FAR = impostores aceitos / total impostores
        far = np.mean(impostor >= tau) * 100
        # FRR = genuínos recusados / total genuínos
        frr = np.mean(genuine < tau) * 100
        rows.append({
            "threshold": tau,
            "FAR_percent": round(far, 2),
            "FRR_percent": round(frr, 2),
        })

    # Salvar CSV de métricas
    csv_path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy_percent", f"{best_acc*100:.2f}"])
        writer.writerow(["best_threshold", f"{best_t:.4f}"])
        writer.writerow(["AUC", f"{roc_auc:.4f}"])
        writer.writerow(["EER_percent", f"{eer*100:.2f}"])
        writer.writerow(["total_pairs", len(labels)])
        writer.writerow(["genuine_pairs", int(np.sum(labels == 1))])
        writer.writerow(["impostor_pairs", int(np.sum(labels == 0))])
        writer.writerow(["genuine_mean_score", f"{np.mean(genuine):.4f}"])
        writer.writerow(["impostor_mean_score", f"{np.mean(impostor):.4f}"])
        for row in rows:
            writer.writerow([f"FAR_tau_{row['threshold']}", row["FAR_percent"]])
            writer.writerow([f"FRR_tau_{row['threshold']}", row["FRR_percent"]])

    print(f"  ✅ Salvo em: {csv_path}")
    print(f"  Acurácia: {best_acc*100:.2f}% | AUC: {roc_auc:.4f} | EER: {eer*100:.2f}%")
    for row in rows:
        print(f"  τ={row['threshold']}: FAR={row['FAR_percent']}%, FRR={row['FRR_percent']}%")

    return rows, best_acc, roc_auc, eer


def generate_consolidated_csv(baseline_rows, personal_results):
    """
    Item 3: Tabela consolidada antes vs depois do ataque.
    """
    print("\n" + "=" * 70)
    print(" 3/3  TABELA CONSOLIDADA: BASELINE vs ATAQUE ".center(70, "═"))
    print("=" * 70)

    # Ler ataque LFW
    lfw_attack_path = os.path.join(RESULTS_DIR, "lfw_swap_attack_results.csv")
    if not os.path.exists(lfw_attack_path):
        print("⚠ lfw_swap_attack_results.csv não encontrado.")
        return

    lfw_attack = pd.read_csv(lfw_attack_path)

    # Calcular ASR do LFW
    n_lfw = len(lfw_attack)
    asr_lfw_045 = lfw_attack["fooled_045"].mean() * 100
    asr_lfw_060 = lfw_attack["fooled_060"].mean() * 100
    sim_mean_lfw = lfw_attack["similarity"].mean()

    # Calcular ASR pessoal
    if personal_results:
        n_pers = len(personal_results)
        asr_pers_045 = sum(r["fooled_045"] for r in personal_results) / n_pers * 100
        asr_pers_060 = sum(r["fooled_060"] for r in personal_results) / n_pers * 100
        sim_mean_pers = np.mean([r["similarity"] for r in personal_results])
    else:
        n_pers = 0
        asr_pers_045 = asr_pers_060 = sim_mean_pers = 0

    # Montar tabela consolidada
    csv_path = os.path.join(RESULTS_DIR, "consolidated_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["indicator", "baseline_no_attack", "attack_personal", "attack_lfw"])

        # FAR baseline
        if baseline_rows:
            far_045 = next(r["FAR_percent"] for r in baseline_rows if r["threshold"] == 0.45)
            far_060 = next(r["FAR_percent"] for r in baseline_rows if r["threshold"] == 0.60)
            frr_045 = next(r["FRR_percent"] for r in baseline_rows if r["threshold"] == 0.45)
            frr_060 = next(r["FRR_percent"] for r in baseline_rows if r["threshold"] == 0.60)
        else:
            far_045 = far_060 = frr_045 = frr_060 = "N/A"

        writer.writerow(["FAR_tau_0.45 (%)", far_045, f"{asr_pers_045:.1f}", f"{asr_lfw_045:.1f}"])
        writer.writerow(["FAR_tau_0.60 (%)", far_060, f"{asr_pers_060:.1f}", f"{asr_lfw_060:.1f}"])
        writer.writerow(["FRR_tau_0.45 (%)", frr_045, "-", "-"])
        writer.writerow(["FRR_tau_0.60 (%)", frr_060, "-", "-"])
        writer.writerow(["mean_similarity", "-", f"{sim_mean_pers:.4f}", f"{sim_mean_lfw:.4f}"])
        writer.writerow(["num_samples", "6000 pairs", f"{n_pers} swaps", f"{n_lfw} swaps"])

    print(f"  ✅ Salvo em: {csv_path}")
    print()
    print(f"  {'Indicador':<25s} | {'Baseline':>10s} | {'Pessoal':>10s} | {'LFW':>10s}")
    print(f"  {'─'*25}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}")
    print(f"  {'FAR τ=0.45':<25s} | {str(far_045):>9s}% | {asr_pers_045:>9.1f}% | {asr_lfw_045:>9.1f}%")
    print(f"  {'FAR τ=0.60':<25s} | {str(far_060):>9s}% | {asr_pers_060:>9.1f}% | {asr_lfw_060:>9.1f}%")
    print(f"  {'Sim. Média':<25s} | {'—':>10s} | {sim_mean_pers:>10.4f} | {sim_mean_lfw:>10.4f}")


def main():
    print("=" * 70)
    print(" 📊 GERADOR DE TABELAS PARA O PAPER ".center(70))
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Carregar modelos uma vez
    print("\n🔧 Carregando modelos (YOLOv8 + AdaFace)...")
    yolo = YOLO(YOLO_FACE_PATH, verbose=False)
    adaface = load_adaface(ADAFACE_CKPT).to(DEVICE)

    # 1. CSV do dataset pessoal
    personal_results = generate_personal_attack_csv(yolo, adaface)

    # 2. CSV de métricas do baseline
    baseline_data = generate_baseline_metrics_csv()
    baseline_rows = baseline_data[0] if baseline_data else None

    # 3. Tabela consolidada
    generate_consolidated_csv(baseline_rows, personal_results)

    print("\n" + "=" * 70)
    print("✅ Todos os CSVs gerados em app/results/:")
    print("   📄 personal_swap_attack_results.csv")
    print("   📄 baseline_metrics.csv")
    print("   📄 consolidated_results.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

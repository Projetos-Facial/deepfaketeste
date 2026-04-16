"""
Experimento de Ataque com Deepfake no LFW (Labeled Faces in the Wild).

Mesmo raciocínio do run_swaps_experiment.py (dataset pessoal), agora aplicado
ao dataset acadêmico LFW:

  1. Seleciona pares de identidades diferentes do LFW
  2. Usa FaceFusion (inswapper_128) para trocar o rosto da Pessoa A na foto da Pessoa B
  3. Compara o swap com a foto original da Pessoa A usando AdaFace
  4. Se a similaridade >= threshold → o deepfake enganou a biometria (ataque bem-sucedido)
  
Gera gráficos e relatórios de FAR/FRR sob ataque para comparação com o baseline.
"""

import os
import math
import sys
import random
import time
import subprocess
import csv
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch

# Adiciona diretório app ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_lfw import (
    ADAFACE_CKPT, YOLO_FACE_PATH, FACE_SIZE, DEVICE,
    LFW_DIR, PAIRS_FILE, MODELS_DIR,
    load_adaface, detect_and_crop_face, preprocess_face, get_embedding,
    ensure_lfw_images, download_file, PAIRS_URL,
    ADAFACE_URL, YOLO_FACE_URL,
)
from ultralytics import YOLO

# ==============================================================================
# Configurações do Experimento
# ==============================================================================
SWAPS_DIR = "data/lfw/swaps"            # Pasta onde as imagens trocadas são salvas
RESULTS_DIR = "app/results"

# Número de swaps a gerar (cada swap = 1 chamada ao FaceFusion)
# 100 swaps para significância estatística adequada (IC 95% < ±10%)
NUM_SWAPS = 100

# Thresholds para avaliar (os mesmos do artigo/baseline)
THRESHOLDS = [0.45, 0.60]

# Mínimo de fotos que uma pessoa precisa ter para ser candidata
# Reduzido para 2 para ampliar a diversidade de identidades
MIN_IMAGES_PER_PERSON = 2

# Seed para reprodutibilidade
RANDOM_SEED = 42


def get_lfw_identities(lfw_dir, min_images=MIN_IMAGES_PER_PERSON):
    """
    Retorna dict {nome_pessoa: [lista de caminhos das fotos]}
    filtrando só quem tem pelo menos `min_images` fotos.
    """
    identities = {}
    for person_name in sorted(os.listdir(lfw_dir)):
        person_dir = os.path.join(lfw_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        imgs = sorted([
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if len(imgs) >= min_images:
            identities[person_name] = imgs
    return identities


def generate_swap_pairs(identities, num_swaps, seed=RANDOM_SEED):
    """
    Gera uma lista de tuplas:
      (source_name, source_path, target_name, target_path)
    
    source = pessoa cujo rosto será "colado" (o atacante tenta se passar por ela)
    target = pessoa cujo corpo/fundo será usado (a foto onde o rosto será substituído)
    
    Garantimos que source e target são SEMPRE pessoas DIFERENTES.
    """
    rng = random.Random(seed)
    names = list(identities.keys())
    pairs = []
    attempts = 0
    
    while len(pairs) < num_swaps and attempts < num_swaps * 10:
        attempts += 1
        src_name = rng.choice(names)
        tgt_name = rng.choice(names)
        
        # Precisam ser pessoas diferentes
        if src_name == tgt_name:
            continue
        
        src_img = rng.choice(identities[src_name])
        tgt_img = rng.choice(identities[tgt_name])
        
        pairs.append((src_name, src_img, tgt_name, tgt_img))
    
    return pairs


def run_facefusion_swap(src_path, tgt_path, out_path, cwd="facefusion"):
    """Roda o FaceFusion para gerar um swap. Retorna True se sucesso."""
    cmd = [
        "python", "facefusion.py", "headless-run",
        "--processors", "face_swapper",
        "--face-swapper-model", "inswapper_128",
        "-s", os.path.abspath(src_path),
        "-t", os.path.abspath(tgt_path),
        "-o", os.path.abspath(out_path),
        "--execution-providers", "cuda"
    ]
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        print(f"      ⚠ Erro no FaceFusion: {e.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"      ⚠ Timeout no FaceFusion (>120s)")
        return False


def main():
    print("=" * 90)
    print("🚀 EXPERIMENTO DE ATAQUE DEEPFAKE NO LFW".center(90))
    print("Face Swap + Avaliação Biométrica (AdaFace)".center(90))
    print("=" * 90)

    # ──────────────────────────────────────────────────────────────
    # PASSO 0: Garantir que os dados e modelos existem
    # ──────────────────────────────────────────────────────────────
    print("\n📦 Passo 0/4: Verificando dados e modelos...")
    ensure_lfw_images()
    download_file(PAIRS_URL, PAIRS_FILE)
    download_file(ADAFACE_URL, ADAFACE_CKPT, is_gdrive=True)
    download_file(YOLO_FACE_URL, YOLO_FACE_PATH)
    os.makedirs(SWAPS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # PASSO 1: Selecionar identidades e gerar pares para swap
    # ──────────────────────────────────────────────────────────────
    print(f"\n🔍 Passo 1/4: Selecionando identidades do LFW (mín. {MIN_IMAGES_PER_PERSON} fotos)...")
    identities = get_lfw_identities(LFW_DIR, MIN_IMAGES_PER_PERSON)
    print(f"   Encontradas {len(identities)} identidades elegíveis.")
    
    swap_pairs = generate_swap_pairs(identities, NUM_SWAPS)
    print(f"   Gerados {len(swap_pairs)} pares de swap para processar.")
    print(f"   Seed: {RANDOM_SEED} (para reprodutibilidade)")

    # Mostra os primeiros pares para referência
    print(f"\n   Primeiros 5 pares:")
    for i, (sn, sp, tn, tp) in enumerate(swap_pairs[:5]):
        print(f"     {i+1}. [Rosto] {sn} → [Corpo] {tn}")

    # ──────────────────────────────────────────────────────────────
    # PASSO 2: Gerar face swaps com FaceFusion
    # ──────────────────────────────────────────────────────────────
    print(f"\n🎭 Passo 2/4: Gerando {len(swap_pairs)} face swaps com FaceFusion...")
    print("   (Cada swap pode levar alguns segundos)\n")
    
    swap_results = []  # (src_name, src_path, tgt_name, tgt_path, out_path, success)
    t0 = time.time()
    
    for i, (src_name, src_path, tgt_name, tgt_path) in enumerate(swap_pairs):
        src_basename = os.path.basename(src_path).split('.')[0]
        tgt_basename = os.path.basename(tgt_path).split('.')[0]
        out_filename = f"swap_{src_basename}_in_{tgt_basename}.jpg"
        out_path = os.path.join(SWAPS_DIR, out_filename)
        
        print(f"   [{i+1:3d}/{len(swap_pairs)}] {src_name} → {tgt_name} ...", end=" ", flush=True)
        
        # Se o swap já existe, não refazer
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print("✅ (já existe)")
            success = True
        else:
            success = run_facefusion_swap(src_path, tgt_path, out_path)
            if success:
                print("✅")
            else:
                print("❌")
        
        swap_results.append((src_name, src_path, tgt_name, tgt_path, out_path, success))
    
    elapsed_swap = time.time() - t0
    n_success = sum(1 for r in swap_results if r[5])
    n_fail = len(swap_results) - n_success
    print(f"\n   Swaps concluídos em {elapsed_swap:.1f}s")
    print(f"   ✅ Sucesso: {n_success} | ❌ Falha: {n_fail}")

    # ──────────────────────────────────────────────────────────────
    # PASSO 3: Avaliar os swaps com AdaFace
    # ──────────────────────────────────────────────────────────────
    print(f"\n🧠 Passo 3/4: Carregando AdaFace + YOLOv8 para avaliação...")
    yolo = YOLO(YOLO_FACE_PATH, verbose=False)
    adaface = load_adaface(ADAFACE_CKPT).to(DEVICE)
    
    print("   Comparando cada swap com a foto original da pessoa-alvo (source)...\n")

    evaluation_results = []
    
    for i, (src_name, src_path, tgt_name, tgt_path, out_path, success) in enumerate(swap_results):
        if not success or not os.path.exists(out_path):
            continue
        
        img_swap = cv2.imread(out_path)
        img_src = cv2.imread(src_path)
        
        if img_swap is None or img_src is None:
            print(f"   [{i+1}] ⚠ Não conseguiu carregar imagens, pulando...")
            continue
        
        # Detectar e extrair embeddings
        face_swap = detect_and_crop_face(yolo, img_swap, FACE_SIZE)
        face_src = detect_and_crop_face(yolo, img_src, FACE_SIZE)
        
        emb_swap = get_embedding(adaface, preprocess_face(face_swap), DEVICE)
        emb_src = get_embedding(adaface, preprocess_face(face_src), DEVICE)
        
        sim = float(np.dot(emb_swap, emb_src) / (np.linalg.norm(emb_swap) * np.linalg.norm(emb_src) + 1e-8))
        
        evaluation_results.append({
            "source_name": src_name,
            "target_name": tgt_name,
            "swap_file": os.path.basename(out_path),
            "source_file": os.path.basename(src_path),
            "target_file": os.path.basename(tgt_path),
            "similarity": sim,
        })
        
        # Visual feedback rápido
        status_045 = "🚨 ENGANOU" if sim >= 0.45 else "🛡️ BARRADO"
        status_060 = "🚨 ENGANOU" if sim >= 0.60 else "🛡️ BARRADO"
        print(f"   [{i+1:3d}] {src_name:25s} → {tgt_name:25s}  "
              f"sim={sim:.4f}  τ0.45:{status_045}  τ0.60:{status_060}")

    # ──────────────────────────────────────────────────────────────
    # PASSO 4: Relatório Final
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(" 📊 RESULTADOS DO ATAQUE DEEPFAKE NO LFW ".center(100, "═"))
    print(f"{'='*100}")
    
    if not evaluation_results:
        print("\n❌ Nenhum swap foi avaliado com sucesso. Verifique se o FaceFusion está funcionando.")
        return
    
    sims = np.array([r["similarity"] for r in evaluation_results])
    n_total = len(sims)
    
    # ── Tabela detalhada ─────────────────────────────────────────
    table_data = []
    for r in evaluation_results:
        sim = r["similarity"]
        table_data.append({
            "Swap (Fraude)": r["swap_file"][:40],
            "Pessoa Alvo": r["source_name"],
            "Corpo Usado": r["target_name"],
            "Similaridade": f"{sim:.4f}",
            "τ=0.45": "🚨 ENGANOU" if sim >= 0.45 else "🛡️ BARRADO",
            "τ=0.60": "🚨 ENGANOU" if sim >= 0.60 else "🛡️ BARRADO",
        })
    
    df = pd.DataFrame(table_data)
    print("\n" + tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    # ── Estatísticas de ataque por threshold ─────────────────────
    print(f"\n{'='*80}")
    print(" 📈 TAXA DE SUCESSO DO ATAQUE (Attack Success Rate) ".center(80, "═"))
    print(f"{'='*80}")
    
    def wilson_ci(n_success, n_total, z=1.96):
        """Intervalo de confiança de Wilson (95%) para proporção."""
        if n_total == 0:
            return 0, 0
        p_hat = n_success / n_total
        denom = 1 + z**2 / n_total
        center = (p_hat + z**2 / (2 * n_total)) / denom
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
        return max(0, center - margin), min(1, center + margin)

    for tau in THRESHOLDS:
        n_fooled = int(np.sum(sims >= tau))
        asr = n_fooled / n_total * 100
        ci_low, ci_high = wilson_ci(n_fooled, n_total)
        print(f"\n   Threshold τ = {tau:.2f}:")
        print(f"     Swaps que ENGANARAM a IA: {n_fooled}/{n_total} ({asr:.1f}%)")
        print(f"     Swaps BARRADOS pela IA:   {n_total - n_fooled}/{n_total} ({100 - asr:.1f}%)")
        print(f"     IC 95% (Wilson):          [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    
    # ── Estatísticas descritivas ─────────────────────────────────
    print(f"\n   Estatísticas de Similaridade dos Swaps:")
    print(f"     Média:   {np.mean(sims):.4f}")
    print(f"     Mediana: {np.median(sims):.4f}")
    print(f"     Desvio:  {np.std(sims):.4f}")
    print(f"     Mín:     {np.min(sims):.4f}")
    print(f"     Máx:     {np.max(sims):.4f}")
    print(f"     Q1 (25%): {np.percentile(sims, 25):.4f}")
    print(f"     Q3 (75%): {np.percentile(sims, 75):.4f}")

    # ── Testes estatísticos ──────────────────────────────────────
    try:
        from scipy import stats as sp_stats
        baseline_csv_stat = os.path.join(RESULTS_DIR, "lfw_scores.csv")
        if os.path.exists(baseline_csv_stat):
            bl_df = pd.read_csv(baseline_csv_stat)
            genuine_sc = bl_df[bl_df["label"] == 1]["score"].values
            impostor_sc = bl_df[bl_df["label"] == 0]["score"].values

            print(f"\n{'='*80}")
            print(" 🔬 TESTES ESTATÍSTICOS ".center(80, "═"))
            print(f"{'='*80}")

            # KS test: swaps vs genuínos
            ks_stat_g, ks_p_g = sp_stats.ks_2samp(sims, genuine_sc)
            print(f"\n   Kolmogorov-Smirnov (Swaps vs Genuínos):")
            print(f"     KS statistic = {ks_stat_g:.4f}")
            print(f"     p-value      = {ks_p_g:.2e}")
            print(f"     {'→ Distribuições DIFERENTES (p < 0.05)' if ks_p_g < 0.05 else '→ Não rejeita H0 (distribuições similares)'}")

            # KS test: swaps vs impostores
            ks_stat_i, ks_p_i = sp_stats.ks_2samp(sims, impostor_sc)
            print(f"\n   Kolmogorov-Smirnov (Swaps vs Impostores):")
            print(f"     KS statistic = {ks_stat_i:.4f}")
            print(f"     p-value      = {ks_p_i:.2e}")
            print(f"     {'→ Distribuições DIFERENTES (p < 0.05)' if ks_p_i < 0.05 else '→ Não rejeita H0 (distribuições similares)'}")

            # Mann-Whitney U: swaps vs genuínos
            u_stat, u_p = sp_stats.mannwhitneyu(sims, genuine_sc, alternative='two-sided')
            print(f"\n   Mann-Whitney U (Swaps vs Genuínos):")
            print(f"     U statistic  = {u_stat:.0f}")
            print(f"     p-value      = {u_p:.2e}")

            # Cohen's d: effect size swaps vs genuínos
            pooled_std = np.sqrt((np.std(sims)**2 + np.std(genuine_sc)**2) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(sims) - np.mean(genuine_sc)) / pooled_std
                print(f"\n   Cohen's d (Swaps vs Genuínos): {cohens_d:.4f}")
                if abs(cohens_d) < 0.2:
                    print(f"     → Efeito NEGLIGÍVEL")
                elif abs(cohens_d) < 0.5:
                    print(f"     → Efeito PEQUENO")
                elif abs(cohens_d) < 0.8:
                    print(f"     → Efeito MÉDIO")
                else:
                    print(f"     → Efeito GRANDE")

            print(f"\n   Resumo: Média Genuínos={np.mean(genuine_sc):.4f}  "
                  f"Média Impostores={np.mean(impostor_sc):.4f}  "
                  f"Média Swaps={np.mean(sims):.4f}")
        else:
            print("\n   ⚠ lfw_scores.csv não encontrado — testes estatísticos pulados.")
    except ImportError:
        print("\n   ⚠ scipy não instalado — testes estatísticos pulados. Instale com: pip install scipy")
    
    # ── Salvar CSV de resultados ─────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "lfw_swap_attack_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "swap_id", "source_name", "target_name",
            "swap_file", "source_file", "target_file",
            "similarity", "fooled_045", "fooled_060"
        ])
        for idx, r in enumerate(evaluation_results):
            writer.writerow([
                idx, r["source_name"], r["target_name"],
                r["swap_file"], r["source_file"], r["target_file"],
                f"{r['similarity']:.6f}",
                int(r["similarity"] >= 0.45),
                int(r["similarity"] >= 0.60),
            ])
    print(f"\n   📁 Resultados detalhados salvos em: {csv_path}")
    
    # ── Gráficos ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # 1. Histograma dos scores de ataque
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(sims, bins=40, alpha=0.7, color="#E91E63", edgecolor="white",
                label=f"Swaps (n={n_total})")
        
        for tau in THRESHOLDS:
            n_fooled = int(np.sum(sims >= tau))
            asr = n_fooled / n_total * 100
            ax.axvline(x=tau, linestyle="--", linewidth=2,
                       label=f"τ={tau} (ASR={asr:.1f}%)")
        
        ax.set_xlabel("Similaridade Cosseno (Swap vs Original)", fontsize=12)
        ax.set_ylabel("Quantidade", fontsize=12)
        ax.set_title(f"Distribuição dos Scores — Ataque Deepfake no LFW (n={n_total})\n"
                      "FaceFusion (inswapper_128) + AdaFace IR-50", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        hist_path = os.path.join(RESULTS_DIR, "lfw_swap_attack_histogram.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"   📊 Histograma de ataque salvo em: {hist_path}")
        
        # 2. Comparação Baseline vs Ataque (se existir o baseline)
        baseline_csv = os.path.join(RESULTS_DIR, "lfw_scores.csv")
        if os.path.exists(baseline_csv):
            baseline_df = pd.read_csv(baseline_csv)
            genuine_scores = baseline_df[baseline_df["label"] == 1]["score"].values
            impostor_scores = baseline_df[baseline_df["label"] == 0]["score"].values
            
            fig2, ax2 = plt.subplots(figsize=(11, 6))
            ax2.hist(genuine_scores, bins=80, alpha=0.55, color="#2196F3", density=True,
                     label=f"Genuínos LFW (n={len(genuine_scores)})")
            ax2.hist(impostor_scores, bins=80, alpha=0.55, color="#4CAF50", density=True,
                     label=f"Impostores LFW (n={len(impostor_scores)})")
            ax2.hist(sims, bins=40, alpha=0.65, color="#F44336", density=True,
                     label=f"Deepfakes/Swaps (n={n_total})")
            
            for tau in THRESHOLDS:
                ax2.axvline(x=tau, linestyle="--", linewidth=1.5, color="black",
                            label=f"τ={tau}")
            
            ax2.set_xlabel("Similaridade Cosseno", fontsize=12)
            ax2.set_ylabel("Densidade", fontsize=12)
            ax2.set_title(f"Baseline vs Ataque Deepfake — LFW (n={n_total} swaps)\n"
                          "Onde os swaps caem na distribuição?", fontsize=13)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            
            comp_path = os.path.join(RESULTS_DIR, "lfw_baseline_vs_attack.png")
            fig2.savefig(comp_path, dpi=150)
            plt.close(fig2)
            print(f"   📊 Comparação baseline vs ataque: {comp_path}")
        
        # 3. Curva ASR vs Threshold (variação contínua)
        tau_range = np.arange(0.05, 0.95, 0.01)
        asr_values = [np.mean(sims >= t) * 100 for t in tau_range]
        ci_low_values = []
        ci_high_values = []
        for t in tau_range:
            n_f = int(np.sum(sims >= t))
            lo, hi = wilson_ci(n_f, n_total)
            ci_low_values.append(lo * 100)
            ci_high_values.append(hi * 100)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(tau_range, asr_values, color="#D32F2F", lw=2.5, label="ASR (%)")
        ax3.fill_between(tau_range, ci_low_values, ci_high_values,
                         alpha=0.2, color="#D32F2F", label="IC 95% (Wilson)")
        
        for tau in THRESHOLDS:
            asr_at = np.mean(sims >= tau) * 100
            ax3.axvline(x=tau, linestyle="--", color="black", alpha=0.6)
            ax3.scatter([tau], [asr_at], color="black", zorder=5, s=60)
            ax3.annotate(f"τ={tau}\nASR={asr_at:.1f}%",
                         xy=(tau, asr_at), xytext=(tau + 0.04, asr_at + 3),
                         fontsize=9, fontweight='bold',
                         arrowprops=dict(arrowstyle="->", color="black"))
        
        ax3.set_xlabel("Threshold (τ)", fontsize=12)
        ax3.set_ylabel("Attack Success Rate — ASR (%)", fontsize=12)
        ax3.set_title(f"ASR vs Threshold — Ataque Deepfake no LFW (n={n_total})\n"
                       "FaceFusion (inswapper_128) + AdaFace IR-50", fontsize=13)
        ax3.set_ylim(-2, 105)
        ax3.legend(fontsize=10, loc="lower left")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        
        asr_path = os.path.join(RESULTS_DIR, "lfw_asr_vs_threshold.png")
        fig3.savefig(asr_path, dpi=150)
        plt.close(fig3)
        print(f"   📊 Curva ASR vs Threshold salva em: {asr_path}")

        # 4. Box plot comparativo (Genuínos vs Swaps vs Impostores)
        if os.path.exists(baseline_csv):
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            box_data = [impostor_scores, sims, genuine_scores]
            box_labels = [f"Impostores\n(n={len(impostor_scores)})",
                          f"Deepfakes\n(n={n_total})",
                          f"Genuínos\n(n={len(genuine_scores)})"]
            bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True,
                             medianprops=dict(color='black', linewidth=2))
            colors = ['#4CAF50', '#F44336', '#2196F3']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for tau in THRESHOLDS:
                ax4.axhline(y=tau, linestyle='--', color='gray', alpha=0.7,
                            label=f'τ={tau}')
            ax4.set_ylabel("Similaridade Cosseno", fontsize=12)
            ax4.set_title("Distribuição de Scores: Impostores vs Deepfakes vs Genuínos", fontsize=13)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
            fig4.tight_layout()
            box_path = os.path.join(RESULTS_DIR, "lfw_boxplot_comparison.png")
            fig4.savefig(box_path, dpi=150)
            plt.close(fig4)
            print(f"   📊 Box plot comparativo salvo em: {box_path}")
        
    except Exception as e:
        print(f"   ⚠ Não foi possível gerar gráficos: {e}")
        import traceback
        traceback.print_exc()

    # ── Conclusão ────────────────────────────────────────────────
    asr_045 = np.mean(sims >= 0.45) * 100
    asr_060 = np.mean(sims >= 0.60) * 100
    n_fooled_045 = int(np.sum(sims >= 0.45))
    n_fooled_060 = int(np.sum(sims >= 0.60))
    ci_lo_045, ci_hi_045 = wilson_ci(n_fooled_045, n_total)
    ci_lo_060, ci_hi_060 = wilson_ci(n_fooled_060, n_total)
    
    conclusion = f"""
{'=' * 70}
CONCLUSÃO DO ATAQUE DEEPFAKE — LFW
{'=' * 70}

Foram gerados {n_total} face swaps utilizando FaceFusion (inswapper_128)
sobre o dataset LFW (Labeled Faces in the Wild).

Cada swap consiste em colar o rosto da Pessoa A (source/alvo) na foto 
da Pessoa B (target/corpo). Depois, comparamos o swap com a foto 
original da Pessoa A usando o modelo AdaFace IR-50.

Se a similaridade >= threshold, o deepfake ENGANOU o sistema biométrico.

  TAXA DE SUCESSO DO ATAQUE (Attack Success Rate):
  
    τ = 0.45: ASR = {asr_045:.1f}% ({n_fooled_045}/{n_total})  IC 95%: [{ci_lo_045*100:.1f}%, {ci_hi_045*100:.1f}%]
    τ = 0.60: ASR = {asr_060:.1f}% ({n_fooled_060}/{n_total})  IC 95%: [{ci_lo_060*100:.1f}%, {ci_hi_060*100:.1f}%]

  Estatísticas de Similaridade:
    Média:   {np.mean(sims):.4f}
    Mediana: {np.median(sims):.4f}
    Desvio:  {np.std(sims):.4f}
    Mín:     {np.min(sims):.4f}
    Máx:     {np.max(sims):.4f}
    Q1 (25%): {np.percentile(sims, 25):.4f}
    Q3 (75%): {np.percentile(sims, 75):.4f}

INTERPRETAÇÃO:
  - ASR alto = O sistema biométrico é VULNERÁVEL a deepfakes
  - ASR baixo = O sistema consegue BARRAR a maioria das manipulações
  
  Comparando com o baseline (fotos originais sem manipulação):
  - Se os swaps caem na mesma região que pares genuínos → sistema em risco
  - Se os swaps ficam abaixo do threshold → sistema robusto nesse cenário

{'=' * 70}
"""
    conclusion_path = os.path.join(RESULTS_DIR, "lfw_swap_attack_conclusion.txt")
    with open(conclusion_path, "w") as f:
        f.write(conclusion)
    print(f"   📝 Conclusão salva em: {conclusion_path}")
    print(conclusion)
    
    print(f"\n{'='*70}")
    print("Experimento finalizado! Arquivos gerados em app/results/:")
    print("  📄 lfw_swap_attack_results.csv      - Detalhes de cada swap")
    print("  📊 lfw_swap_attack_histogram.png     - Distribuição dos scores")
    print("  📊 lfw_baseline_vs_attack.png        - Baseline vs Ataque")
    print("  📊 lfw_asr_vs_threshold.png          - Curva ASR vs Threshold")
    print("  📊 lfw_boxplot_comparison.png        - Box plot comparativo")
    print("  📝 lfw_swap_attack_conclusion.txt    - Conclusão do experimento")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

"""
Baseline de Verificação Facial no LFW usando YOLOv8 + AdaFace (CVLFace).

Pipeline (usando yolo8face_adaface):
  1. YOLOv8-Face: Detecção e recorte de faces
  2. AdaFace (CVLFace IR-50 via HuggingFace): Extração de embeddings faciais
  3. Similaridade cosseno para verificação
  4. Avaliação: Acurácia, AUC, EER no protocolo padrão do LFW
"""

import os
import sys
import time
import urllib.request
import tarfile
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, accuracy_score

# ==============================================================================
# Adiciona o diretório da nova API ao path
# ==============================================================================
YOLO8FACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolo8face_adaface")
sys.path.insert(0, YOLO8FACE_DIR)

from src import AdaFaceVerifier
from src.utils import preprocess_image, preprocess_batch, get_similarity

# ==============================================================================
# Configurações
# ==============================================================================
DATA_DIR = "data/lfw"
LFW_DIR = os.path.join(DATA_DIR, "lfw_home", "lfw_funneled")
PAIRS_URL = "https://raw.githubusercontent.com/davidsandberg/facenet/master/data/pairs.txt"
PAIRS_FILE = os.path.join(DATA_DIR, "pairs.txt")
MODELS_DIR = "data/models"

# Mantemos as variáveis legadas para compatibilidade com scripts que importam
ADAFACE_CKPT = os.path.join(MODELS_DIR, "adaface_ir50_webface4m.ckpt")
YOLO_FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
YOLO_FACE_PATH = os.path.join(YOLO8FACE_DIR, "model_weights", "yolov8n-face.pt")
ADAFACE_URL = "1HdW-F1GxJv0MVBUIVpE6HAZ3S9SLsytL"

FACE_SIZE = (112, 112)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config path for the new verifier
_CONFIG_PATH = os.path.join(YOLO8FACE_DIR, "configs", "config.yaml")

# ==============================================================================
# Singleton: Carrega o AdaFaceVerifier uma única vez
# ==============================================================================
_verifier_instance = None

def _get_verifier():
    """Retorna instância singleton do AdaFaceVerifier."""
    global _verifier_instance
    if _verifier_instance is None:
        print("   🚀 Inicializando AdaFaceVerifier (CVLFace via HuggingFace)...")
        # Precisamos estar no diretório da API para que os paths relativos funcionem
        old_cwd = os.getcwd()
        os.chdir(YOLO8FACE_DIR)
        try:
            _verifier_instance = AdaFaceVerifier(_CONFIG_PATH)
        finally:
            os.chdir(old_cwd)
        print(f"   ✅ Modelo carregado no device: {_verifier_instance.device}")
    return _verifier_instance


# ==============================================================================
# Downloads (mantidos para compatibilidade)
# ==============================================================================
def download_file(url_or_id, dest, is_gdrive=False):
    if os.path.exists(dest):
        print(f"   Já existe: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if is_gdrive:
        # Com a nova API, o modelo é baixado automaticamente do HuggingFace.
        # Mas mantemos a interface caso algum script antigo ainda chame.
        print(f"   ⚠ Modelo legado (gdown) não é mais necessário. Usando CVLFace/HuggingFace.")
        return
    else:
        print(f"   Baixando: {url_or_id}")
        urllib.request.urlretrieve(url_or_id, dest)
        print(f"   Salvo em: {dest}")


def ensure_lfw_images():
    """Garante que as imagens LFW estão disponíveis."""
    if os.path.isdir(LFW_DIR) and len(os.listdir(LFW_DIR)) > 100:
        print(f"   Imagens LFW encontradas em: {LFW_DIR}")
        return
    tgz_path = os.path.join(DATA_DIR, "lfw-funneled.tgz")
    lfw_home = os.path.join(DATA_DIR, "lfw_home")
    os.makedirs(lfw_home, exist_ok=True)
    download_file("https://web.archive.org/web/20230225141014if_/http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz", tgz_path)
    print("   Extraindo...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=lfw_home)
    print("   Extração concluída!")


# ==============================================================================
# Parsing do pairs.txt (protocolo padrão LFW)
# ==============================================================================
def parse_pairs(pairs_file, lfw_dir):
    """Retorna lista de (img1_path, img2_path, is_same)."""
    pairs = []
    with open(pairs_file, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split("\t")
    n_folds = int(header[0])
    n_pairs = int(header[1])
    idx = 1

    for fold in range(n_folds):
        # Pares positivos (mesma pessoa)
        for _ in range(n_pairs):
            parts = lines[idx].strip().split("\t")
            name = parts[0]
            i1, i2 = int(parts[1]), int(parts[2])
            p1 = os.path.join(lfw_dir, name, f"{name}_{i1:04d}.jpg")
            p2 = os.path.join(lfw_dir, name, f"{name}_{i2:04d}.jpg")
            pairs.append((p1, p2, True))
            idx += 1

        # Pares negativos (pessoas diferentes)
        for _ in range(n_pairs):
            parts = lines[idx].strip().split("\t")
            n1, i1 = parts[0], int(parts[1])
            n2, i2 = parts[2], int(parts[3])
            p1 = os.path.join(lfw_dir, n1, f"{n1}_{i1:04d}.jpg")
            p2 = os.path.join(lfw_dir, n2, f"{n2}_{i2:04d}.jpg")
            pairs.append((p1, p2, False))
            idx += 1

    return pairs


# ==============================================================================
# Funções de Compatibilidade (exportadas para run_swaps_lfw.py e outros)
# ==============================================================================
def load_adaface(checkpoint_path=None, architecture=None):
    """
    Compatibilidade: retorna o modelo da nova API.
    O checkpoint_path é ignorado — usamos HuggingFace.
    """
    verifier = _get_verifier()
    return verifier.model


def detect_and_crop_face(yolo_model, img_bgr, target_size=(112, 112)):
    """
    Detecta face com YOLOv8 e retorna a face recortada como RGB.
    Usa o detector interno da nova API para consistência.
    """
    from src.utils import detector as yolo_detector
    
    results = yolo_detector(img_bgr, verbose=False)[0]
    if len(results.boxes) > 0:
        box = results.boxes.xyxy[0].cpu().numpy().astype(int)
        face_img = img_bgr[max(0, box[1]):min(img_bgr.shape[0], box[3]),
                           max(0, box[0]):min(img_bgr.shape[1], box[2])]
        face_img = cv2.resize(face_img, target_size)
    else:
        face_img = cv2.resize(img_bgr, target_size)

    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return face_rgb


def preprocess_face(face_rgb):
    """Normaliza face RGB (np array) para tensor AdaFace."""
    face = cv2.resize(face_rgb, FACE_SIZE).astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5  # [-1, 1]
    face = np.transpose(face, (2, 0, 1))  # CHW
    return torch.FloatTensor(face).unsqueeze(0)


@torch.no_grad()
def get_embedding(model, face_tensor, device):
    """Extrai embedding normalizado usando o modelo CVLFace."""
    face_tensor = face_tensor.to(device)
    output = model(face_tensor)
    # CVLFace retorna tensor direto
    if isinstance(output, tuple):
        embedding = output[0]
    else:
        embedding = output
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy().flatten()


# ==============================================================================
# Avaliação
# ==============================================================================
def find_best_threshold(similarities, labels):
    best_acc, best_t = 0, 0
    for t in np.arange(0.0, 1.0, 0.001):
        preds = (np.array(similarities) >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 60)
    print("BASELINE — Verificação Facial no LFW")
    print("YOLOv8-Face + AdaFace CVLFace IR-50 (via HuggingFace)")
    print("=" * 60)

    # 1. Downloads
    print("\n[1/5] Preparando dados e modelos...")
    ensure_lfw_images()
    download_file(PAIRS_URL, PAIRS_FILE)
    # O modelo AdaFace agora é baixado automaticamente do HuggingFace
    # Não precisa mais de gdown nem de download manual

    # 2. Carregar modelos (via nova API)
    print(f"\n[2/5] Carregando modelos (device: {DEVICE})...")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    verifier = _get_verifier()
    adaface = verifier.model

    # 3. Carregar pares
    print("\n[3/5] Carregando pares LFW...")
    pairs = parse_pairs(PAIRS_FILE, LFW_DIR)
    n_pos = sum(1 for _, _, s in pairs if s)
    n_neg = len(pairs) - n_pos
    print(f"   Total: {len(pairs)} | Positivos: {n_pos} | Negativos: {n_neg}")

    # 4. Processar pares
    print("\n[4/5] Processando pares...")
    similarities, labels, errors = [], [], 0
    t0 = time.time()

    for i, (p1, p2, is_same) in enumerate(pairs):
        if not os.path.exists(p1) or not os.path.exists(p2):
            errors += 1
            continue

        img1, img2 = cv2.imread(p1), cv2.imread(p2)
        if img1 is None or img2 is None:
            errors += 1
            continue

        # Usar o verifier.verify() da nova API — muito mais rápido
        _, sim = verifier.verify(img1, img2)

        similarities.append(sim)
        labels.append(1 if is_same else 0)

        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(pairs) - i - 1)
            print(f"   {i+1}/{len(pairs)} ({el:.0f}s, ETA: {eta:.0f}s)")

    total_time = time.time() - t0
    print(f"   Concluído em {total_time:.1f}s")
    if errors:
        print(f"   ⚠ {errors} pares ignorados (imagens ausentes)")

    # 5. Avaliação completa
    print("\n[5/5] Avaliação e exportação de resultados")
    print("=" * 60)

    RESULTS_DIR = "app/results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sims  = np.array(similarities)
    labs  = np.array(labels)

    # ── Melhor threshold (grid search) ───────────────────────────
    best_t, best_acc = find_best_threshold(sims, labs)
    fpr, tpr, roc_thresholds = roc_curve(labs, sims)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

    # ── Acurácia por fold ────────────────────────────────────────
    N_FOLDS = 10
    fold_size = len(labs) // N_FOLDS
    fold_accs = []
    for f in range(N_FOLDS):
        s, e = f * fold_size, (f + 1) * fold_size
        fp = (sims[s:e] >= best_t).astype(int)
        fold_accs.append(accuracy_score(labs[s:e], fp))

    # ── Imprimir resumo ──────────────────────────────────────────
    print(f"\n   RESULTADOS NO LFW")
    print(f"   {'─' * 40}")
    print(f"   Pares avaliados:    {len(labs)}")
    print(f"   Melhor threshold:   {best_t:.4f}")
    print(f"   Acurácia:           {best_acc*100:.2f}%")
    print(f"   AUC:                {roc_auc:.4f}")
    print(f"   EER:                {eer*100:.2f}%")
    print(f"   Tempo total:        {total_time:.1f}s")
    print(f"\n   Acurácia por fold:")
    for f, fa in enumerate(fold_accs):
        print(f"     Fold {f+1:2d}: {fa*100:.2f}%")
    print(f"     {'─' * 30}")
    print(f"     Média: {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%")

    # ════════════════════════════════════════════════════════════
    # PASSO A — Salvar scores por par (lfw_scores.csv)
    # ════════════════════════════════════════════════════════════
    import csv
    scores_path = os.path.join(RESULTS_DIR, "lfw_scores.csv")
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "label", "score"])
        for idx, (lab, sim) in enumerate(zip(labs, sims)):
            writer.writerow([idx, int(lab), f"{sim:.6f}"])
    print(f"\n   [A] Scores salvos em: {scores_path}")

    # ════════════════════════════════════════════════════════════
    # PASSO B — FAR e FRR nos thresholds do artigo
    # ════════════════════════════════════════════════════════════
    PAPER_THRESHOLDS = [0.45, 0.60]
    genuine_scores   = sims[labs == 1]
    impostor_scores  = sims[labs == 0]

    print(f"\n   [B] FAR / FRR nos thresholds do artigo")
    print(f"   {'Threshold':>12} | {'FAR (%)':>10} | {'FRR (%)':>10}")
    print(f"   {'─'*12}-+-{'─'*10}-+-{'─'*10}")

    threshold_results = []
    for tau in PAPER_THRESHOLDS:
        far = np.mean(impostor_scores >= tau) * 100   # impostores aceitos
        frr = np.mean(genuine_scores  <  tau) * 100   # genuínos rejeitados
        threshold_results.append((tau, far, frr))
        print(f"   {tau:>12.2f} | {far:>10.2f} | {frr:>10.2f}")

    # ════════════════════════════════════════════════════════════
    # PASSO C — Histograma de distribuição dos scores
    # ════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(genuine_scores,  bins=80, alpha=0.65, color="#2196F3",
            label=f"Genuínos  (n={len(genuine_scores)})", density=True)
    ax.hist(impostor_scores, bins=80, alpha=0.65, color="#F44336",
            label=f"Impostores (n={len(impostor_scores)})", density=True)

    for tau, far, frr in threshold_results:
        ax.axvline(x=tau, linestyle="--", linewidth=1.5,
                   label=f"τ={tau} (FAR={far:.1f}%, FRR={frr:.1f}%)")

    ax.set_xlabel("Similaridade Cosseno", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.set_title("Distribuição dos Scores — LFW Baseline\nYOLOv8-Face + AdaFace CVLFace IR-50", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    hist_path = os.path.join(RESULTS_DIR, "score_hist_baseline.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"\n   [C] Histograma salvo em: {hist_path}")

    # Curva ROC
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"AdaFace CVLFace IR-50 (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax2.scatter([eer], [1 - eer], color="red", zorder=5,
                label=f"EER = {eer*100:.2f}%")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.set_title("Curva ROC — LFW\nYOLOv8-Face + AdaFace CVLFace IR-50")
    ax2.legend(loc="lower right"); ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "roc_curve_lfw.png")
    fig2.savefig(roc_path, dpi=150)
    plt.close(fig2)
    print(f"   [C] Curva ROC salva em: {roc_path}")

    # ════════════════════════════════════════════════════════════
    # PASSO D — Configuração experimental
    # ════════════════════════════════════════════════════════════
    import platform

    config_lines = [
        "=" * 55,
        "CONFIGURAÇÃO EXPERIMENTAL — LFW BASELINE",
        "=" * 55,
        f"Target system    : YOLOv8-Face + AdaFace CVLFace IR-50",
        f"Model source     : HuggingFace (minchul/cvlface_adaface_ir50_ms1mv2)",
        f"Dataset/Protocol : LFW (Labeled Faces in the Wild)",
        f"Thresholds usados: 0.45 e 0.60 (conforme artigo)",
        f"Hardware         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"Device           : {'CUDA' if torch.cuda.is_available() else 'CPU'}",
        f"",
        f"Total de pares   : {len(labs)}",
        f"Positivos        : {int(labs.sum())}",
        f"Negativos        : {len(labs) - int(labs.sum())}",
        f"Pares ignorados  : {errors}",
        f"",
        f"OS               : {platform.system()} {platform.release()}",
        f"Python           : {platform.python_version()}",
        f"PyTorch          : {torch.__version__}",
        f"",
        "=" * 55,
        "RESULTADOS",
        "=" * 55,
        f"Melhor threshold : {best_t:.4f}",
        f"Acurácia         : {best_acc*100:.2f}%",
        f"AUC              : {roc_auc:.4f}",
        f"EER              : {eer*100:.2f}%",
        f"Tempo total (s)  : {total_time:.1f}",
        f"",
        "Acurácia por fold:",
    ]
    for f, fa in enumerate(fold_accs):
        config_lines.append(f"  Fold {f+1:2d}         : {fa*100:.2f}%")
    config_lines.append(f"  Média          : {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%")
    config_lines += [
        f"",
        "=" * 55,
        "FAR / FRR (THRESHOLDS DO ARTIGO)",
        "=" * 55,
        f"{'Threshold':>12} | {'FAR (%)':>10} | {'FRR (%)':>10}",
        f"{'─'*12}-+-{'─'*10}-+-{'─'*10}",
    ]
    for tau, far, frr in threshold_results:
        config_lines.append(f"{tau:>12.2f} | {far:>10.2f} | {frr:>10.2f}")

    config_path = os.path.join(RESULTS_DIR, "experimental_config.txt")
    with open(config_path, "w") as f:
        f.write("\n".join(config_lines) + "\n")
    print(f"   [D] Config experimental salva em: {config_path}")

    # ════════════════════════════════════════════════════════════
    # PASSO E — Conclusão do baseline
    # ════════════════════════════════════════════════════════════
    far_045, frr_045 = threshold_results[0][1], threshold_results[0][2]
    far_060, frr_060 = threshold_results[1][1], threshold_results[1][2]

    conclusion = f"""
{'=' * 55}
CONCLUSÃO DO BASELINE — LFW Face Verification
{'=' * 55}

O pipeline de verificação facial composto por YOLOv8-Face
(detecção) + AdaFace CVLFace IR-50 (embedding via HuggingFace)
apresentou desempenho forte em condições normais no protocolo LFW:

  Acurácia : {best_acc*100:.2f}%  (threshold ótimo = {best_t:.4f})
  AUC      : {roc_auc:.4f}
  EER      : {eer*100:.2f}%

FAR/FRR nos thresholds definidos pelo artigo:

  τ = 0.45 → FAR = {far_045:.2f}%,  FRR = {frr_045:.2f}%
  τ = 0.60 → FAR = {far_060:.2f}%,  FRR = {frr_060:.2f}%

O baseline servirá como referência para avaliar a degradação
do sistema em cenários com manipulações (deepfakes). As duas
distribuições de scores (genuínos vs. impostores) apresentam
separação visível, o que é esperado para um modelo pré-treinado
em larga escala como o AdaFace (MS1MV2).

Os thresholds τ = 0.45 e τ = 0.60 serão usados nas próximas
análises para quantificar o impacto das manipulações faciais
sobre FAR e FRR do sistema.
{'=' * 55}
"""
    conclusion_path = os.path.join(RESULTS_DIR, "baseline_conclusion.txt")
    with open(conclusion_path, "w") as f:
        f.write(conclusion)
    print(f"   [E] Conclusão salva em: {conclusion_path}")
    print(conclusion)

    print(f"\n{'=' * 60}")
    print("Baseline finalizado! Resultados em: app/results/")
    print("  lfw_scores.csv")
    print("  score_hist_baseline.png")
    print("  roc_curve_lfw.png")
    print("  experimental_config.txt")
    print("  baseline_conclusion.txt")


if __name__ == "__main__":
    main()

import os
import sys
import subprocess
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch

# Puxando as funções da pasta pai (app/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline_lfw import ADAFACE_CKPT, YOLO_FACE_PATH, FACE_SIZE, DEVICE, load_adaface, detect_and_crop_face, preprocess_face, get_embedding
from ultralytics import YOLO

def main():
    print("="*90)
    print("🚀 EXPERIMENTO COMPLETO AUTOMATIZADO: FACE SWAP E AVALIAÇÃO".center(90))
    print("="*90)
    
    # Garantir a pasta
    os.makedirs("data/dataset_pessoal/swaps", exist_ok=True)
    
    # Nomes dos arquivos de originais (quem empresta o rosto, quem empresta o corpo)
    # Você pode EDITAR essa lista sempre que quiser rodar com outras fotos que você jogar lá na pasta
    duos = [
        ("pauloserioescuro.jpg", "joaoserio.jpg"),          # 1: Rosto do Paulo,  Corpo do Joao
        ("lucasserioclaro.jpg", "carolserioclaro.jpg"),     # 2: Rosto do Lucas,  Corpo da Carol
        ("carolserioescuro.jpg", "paulosorrindoclaro.jpg"), # 3: Rosto da Carol,  Corpo do Paulo
        ("mateussorrindoclaro.jpg", "jonesserio.jpg"),      # 4: Rosto do Mateus, Corpo do Jones
        ("mariasorrindoclaro.jpeg", "victorserioclaro.jpeg")# 5: Rosto da Maria,  Corpo do Victor
    ]

    print("\n🔧 Passo 1/3: Iniciando FaceFusion (Pode demorar alguns minutos para os Swaps)...")
    
    """ 
    Se o FaceFusion não rodar, certifique-se de instalar os requirements dele antes:
    pip install -r facefusion/requirements.txt
    """
    
    for (src, tgt) in duos:
        src_path = f"data/dataset_pessoal/originais/{src}"
        tgt_path = f"data/dataset_pessoal/originais/{tgt}"
        
        # Pega a extensão exata (jpg ou jpeg) do corpo base para o Facefusion não reclamar
        ext = tgt.split('.')[-1]
        out_path = f"data/dataset_pessoal/swaps/swap_{src.split('.')[0]}_in_{tgt.split('.')[0]}.{ext}"
        
        print(f"\n🎭 Trocando: [Rosto] {src}  -->  [Corpo] {tgt} ...")
        cmd = [
            "python", "facefusion.py", "headless-run",
            "--processors", "face_swapper",
            "--face-swapper-model", "inswapper_128",
            "-s", os.path.abspath(src_path), "-t", os.path.abspath(tgt_path), "-o", os.path.abspath(out_path),
            "--execution-providers", "cuda"
        ]
        try:
            # Removi o bloqueio de prints (DEVNULL) para que possamos ver o erro real!
            result = subprocess.run(cmd, cwd="facefusion", check=True, capture_output=True, text=True)
            print("   ✅ Swap Concluído.")
        except subprocess.CalledProcessError as e:
            print("   ❌ Erro neste swap. Veja os bastidores do que o FaceFusion reclamou:")
            print("--------------------------------------------------")
            print(e.stderr)
            print("--------------------------------------------------")

    print("\n🔧 Passo 2/3: Carregando Modelo ADAFace para Auditar os Swaps...")
    yolo = YOLO(YOLO_FACE_PATH, verbose=False)
    adaface = load_adaface(ADAFACE_CKPT).to(DEVICE)
    
    results = []
    threshold = 0.45
    
    print("\n📊 Passo 3/3: Validando Similaridade (SWAP vs Original)...")
    for (src, tgt) in duos:
        ext = tgt.split('.')[-1]
        out_path = f"data/dataset_pessoal/swaps/swap_{src.split('.')[0]}_in_{tgt.split('.')[0]}.{ext}"
        if not os.path.exists(out_path): continue
            
        src_path = f"data/dataset_pessoal/originais/{src}"
        
        # Extraindo Features
        img_swap = cv2.imread(out_path)
        img_src = cv2.imread(src_path)
        
        face_swap = detect_and_crop_face(yolo, img_swap, FACE_SIZE)
        face_src = detect_and_crop_face(yolo, img_src, FACE_SIZE)
        
        emb_swap = get_embedding(adaface, preprocess_face(face_swap), DEVICE)
        emb_src = get_embedding(adaface, preprocess_face(face_src), DEVICE)
        
        sim = float(np.dot(emb_swap, emb_src) / (np.linalg.norm(emb_swap) * np.linalg.norm(emb_src) + 1e-8))
        match_str = "🚨 FALHOU NA IA (Enganou!)" if sim >= threshold else "🛡️ BARRADO"
        
        results.append({
            "Foto Fraude (Swap)": out_path.split("/")[-1],
            "Alvo da Tentativa (Pessoa Verdadeira)": src,
            "Similaridade de Rosto": f"{sim:.4f}",
            "Threshold de Corte": threshold,
            "O Deepfake Enganou a IA?": match_str
        })
        
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*100)
        print(" 📊 RESULTADOS DO ATAQUE COM DEEPFAKE (SWAP vs ORIGINAL MANIPULADO) ".center(100, "═"))
        print("="*100)
        print("Neste teste criamos a fraude, e tentamos fazer o modelo aprovar como se fosse a pessoa real.")
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
        print("="*100 + "\n")
    else:
        print("\nOs swaps não foram gerados. Você já instalou o FaceFusion? Rode: pip install -r facefusion/requirements.txt")

if __name__ == "__main__":
    main()

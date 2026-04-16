import os
import itertools
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline_lfw import ADAFACE_CKPT, YOLO_FACE_PATH, FACE_SIZE, DEVICE, load_adaface, detect_and_crop_face, preprocess_face, get_embedding
from ultralytics import YOLO

def main():
    originais_dir = "data/dataset_pessoal/originais"
    imgs = sorted([f for f in os.listdir(originais_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(imgs) < 2:
        print("❌ Poucas imagens encontradas.")
        return

    print("🔧 Carregando Motor de IA do Projeto (AdaFace + YOLOv8) ...")
    yolo = YOLO(YOLO_FACE_PATH, verbose=False)
    adaface = load_adaface(ADAFACE_CKPT).to(DEVICE)
    
    embeddings = {}
    print("\n📸 Extraindo embeddings faciais das imagens originais...")
    for img_name in imgs:
        path = os.path.join(originais_dir, img_name)
        img = cv2.imread(path)
        if img is None: 
            continue
        face = detect_and_crop_face(yolo, img, FACE_SIZE)
        emb = get_embedding(adaface, preprocess_face(face), DEVICE)
        embeddings[img_name] = emb
        print(f"   [+] Processado: {img_name}")

    results = []
    threshold = 0.45
    
    # Criaremos testes lógicos entre todas as fotos da sua pasta!
    pairs_to_test = list(itertools.combinations(embeddings.keys(), 2))
    
    print(f"\n📊 Processando {len(pairs_to_test)} combinações de duplas!")
    for imgA, imgB in pairs_to_test:
        embA = embeddings[imgA]
        embB = embeddings[imgB]
        
        sim = float(np.dot(embA, embB) / (np.linalg.norm(embA) * np.linalg.norm(embB) + 1e-8))
        match_str = "👍 SIM" if sim >= threshold else "👎 NÃO"
        
        results.append({
            "Foto Original A": imgA,
            "Foto Original B": imgB,
            "Similaridade": f"{sim:.4f}",
            "Threshold": threshold,
            "O Modelo Achou Igual?": match_str
        })
        
    df = pd.DataFrame(results)
    
    # Salvar resultados completos
    out_csv = "data/dataset_pessoal/resultados_dataset_pessoal.csv"
    df.to_csv(out_csv, index=False)
    
    # Imprimos as correlações mais altas e mais baixas (Top 15 mais confiantes)
    print("\n" + "="*90)
    print(" 📊 RESULTADOS DO SEU DATASET PESSOAL (ORIGINAL vs ORIGINAL) ".center(90, "═"))
    print("="*90)
    print("A tabela abaixo mostra um resumo das maiores combinações identificadas:")
    df["numeric_sim"] = df["Similaridade"].astype(float)
    df_sorted = df.sort_values(by="numeric_sim", ascending=False).drop(columns=["numeric_sim"]).head(20)
    print(tabulate(df_sorted, headers='keys', tablefmt='fancy_grid', showindex=False))
    print("="*90 + "\n")
    print(f"✅ O Relatório de TODAS as fotos com todos da rede gerou o arquivo: {out_csv}")

if __name__ == "__main__":
    main()

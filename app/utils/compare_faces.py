import sys
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

def compare_images(img1_path, img2_path):
    print("\n🔧 Carregando IA...")
    yolo = YOLO(YOLO_FACE_PATH, verbose=False)
    adaface = load_adaface(ADAFACE_CKPT).to(DEVICE)

    print(f"\n📸 Comparando:\n - [A] {img1_path}\n - [B] {img2_path}")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("\n❌ Erro: Imagem não encontrada! Verifique o caminho.")
        return

    # Processamento
    face1 = detect_and_crop_face(yolo, img1, FACE_SIZE)
    face2 = detect_and_crop_face(yolo, img2, FACE_SIZE)

    emb1 = get_embedding(adaface, preprocess_face(face1), DEVICE)
    emb2 = get_embedding(adaface, preprocess_face(face2), DEVICE)

    # Cálculo final
    sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
    threshold = 0.45
    match_str = "👍 SIM" if sim >= threshold else "👎 NÃO"

    # Montando a Tabela final
    data = {
        "Imagem 1": [img1_path.split('/')[-1]],
        "Imagem 2": [img2_path.split('/')[-1]],
        "Score Similaridade": [f"{sim:.4f}"],
        "Corte (Threshold)": [threshold],
        "São a mesma pessoa?": [match_str]
    }
    
    df = pd.DataFrame(data)
    print("\n" + "="*80)
    print(" 📊 RESULTADO DA VERIFICAÇÃO ADAFACE ".center(80, "═"))
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Uso Incorreto!\nSintaxe: python app/compare_faces.py caminho/para/imgA.jpg caminho/para/imgB.jpg")
    else:
        compare_images(sys.argv[1], sys.argv[2])

import os
import glob
import numpy as np
import pandas as pd
import torch
import cv2
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Importamos as funções do LFW para o nosso novo baseline
from baseline_lfw import load_adaface, preprocess_face, get_embedding, DEVICE, ADAFACE_CKPT

YTF_DATA_DIR = "data/ytf/aligned_images_DB"
SPLITS_PATH = "data/ytf/splits.txt"
RESULTS_DIR = "results"
NUM_PAIRS_TO_TEST = 200 # Usamos 200 para rodar rápido inicialmente (podemos aumentar depois)
FRAMES_TO_SAMPLE = 5 # Avalia 5 frames de cada vídeo para tirar a média da identidade

def parse_splits(filepath, max_pairs=None):
    pairs = []
    if not os.path.exists(filepath):
        print(f"❌ Erro: '{filepath}' não encontrado.")
        return pairs
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines[1:]: # pula o cabeçalho
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            # Ex: 1, 1, Sadie_Frost/1, Sadie_Frost/5, 1
            vid1 = parts[2]
            vid2 = parts[3]
            is_same = int(parts[4])
            pairs.append((vid1, vid2, is_same))
            
    if max_pairs:
        # Pega metade autênticos e metade impostores para equilibrar a avaliação!
        genuine = [p for p in pairs if p[2] == 1]
        impostors = [p for p in pairs if p[2] == 0]
        half = max_pairs // 2
        pairs = genuine[:half] + impostors[:half]
        
    print(f"📂 Lidos {len(pairs)} pares de vídeos de YTF (Genuínos: {sum(1 for p in pairs if p[2]==1)}, Impostores: {sum(1 for p in pairs if p[2]==0)}).")
    return pairs

def get_video_mean_embedding(video_rel_path, model):
    video_dir = os.path.join(YTF_DATA_DIR, video_rel_path)
    if not os.path.exists(video_dir):
        return None
        
    # Pega todos os jpgs ordenados
    frames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if len(frames) == 0:
        return None
        
    # Amostragem temporal (pega N frames espaçados do clipe do ator)
    if len(frames) > FRAMES_TO_SAMPLE:
        idx = np.linspace(0, len(frames) - 1, FRAMES_TO_SAMPLE).astype(int)
        sampled_frames = [frames[i] for i in idx]
    else:
        sampled_frames = frames
        
    embeddings = []
    for frame_path in sampled_frames:
        img = cv2.imread(frame_path)
        if img is None: continue
        
        # Como as imagens do YTF já foram alinhadas pela universidade (Crop), 
        # Não precisamos passar pela YOLOv8 para cortar:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (112, 112))
        
        # Adequar para a AdaFace
        face_tensor = preprocess_face(img_resized)
        if face_tensor is not None:
            emb = get_embedding(model, face_tensor, DEVICE)
            embeddings.append(emb)
            
    if len(embeddings) > 0:
        # Tira a MÉDIA do rosto (Fusion level)
        mean_emb = np.mean(embeddings, axis=0)
        # Normaliza o vetor novamente (L2)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        return mean_emb
        
    return None

def compute_similarity(emb1, emb2):
    return float(np.dot(np.squeeze(emb1), np.squeeze(emb2)))

def main():
    print("🔥 Iniciando Baseline do YouTubeFaces DB (Vídeo-to-Vídeo)...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Carrega AdaFace
    adaface = load_adaface(ADAFACE_CKPT).to(DEVICE)
    
    # Carrega Lista de Luta (Video A vs Video B)
    pairs = parse_splits(SPLITS_PATH, max_pairs=NUM_PAIRS_TO_TEST)
    
    results = []
    
    for idx, (vid1, vid2, is_same) in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] YTF: {vid1} vs {vid2}")
        
        emb1 = get_video_mean_embedding(vid1, adaface)
        emb2 = get_video_mean_embedding(vid2, adaface)
        
        if emb1 is None or emb2 is None:
            print(f"   ⚠️ Pulado: Diretórios não encontrados ou vazios.")
            continue
            
        sim = compute_similarity(emb1, emb2)
        results.append({
            "video1": vid1,
            "video2": vid2,
            "is_same": is_same,
            "similarity": sim
        })
        print(f"   Score: {sim:.4f} | Real: {'Autêntico' if is_same else 'Impostor'}")
        
    # Avaliação de Dados
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/ytf_baseline_results.csv", index=False)
    
    true_labels = df['is_same'].values
    scores = df['similarity'].values
    
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    print("\n======================================")
    print("🏆 RESULTADOS DO BASELINE YTF (VÍDEO)")
    print("======================================")
    print(f"Pares Avaliados: {len(df)}")
    print(f"AUC (Acurácia Roc): {roc_auc:.4f}")
    
    genuine_scores = df[df['is_same'] == 1]['similarity'].values
    impostor_scores = df[df['is_same'] == 0]['similarity'].values
    
    print(f"Média Genuínos (Video igual):   {np.mean(genuine_scores):.4f}")
    print(f"Média Impostores (Video dif):   {np.mean(impostor_scores):.4f}")
    print("======================================\n")

if __name__ == "__main__":
    main()

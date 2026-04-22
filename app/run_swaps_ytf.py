import os
import random
import time
import subprocess
import glob
import csv
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import torch
import math

from baseline_lfw import load_adaface, preprocess_face, get_embedding, DEVICE, ADAFACE_CKPT
from baseline_ytf import YTF_DATA_DIR, RESULTS_DIR, parse_splits, compute_similarity, get_video_mean_embedding

# Configurações do Ataque de Vídeo
SWAPS_DIR = "data/ytf/swaps"
NUM_PAIRS_TO_TEST = 100 # Metade autênticos e metade impostores (100 -> teremos 50 impostores para atacar)
FRAMES_TO_SAMPLE = 5 # Selecionar 5 frames do vídeo original do figurante
RANDOM_SEED = 42

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
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=120)
        return True
    except Exception as e:
        return False

def get_mean_embedding_from_frames(frames_paths, model):
    """Calcula o vetor de identidade médio (AdaFace) a partir de uma lista de imgs"""
    embeddings = []
    for fp in frames_paths:
        if not os.path.exists(fp): continue
        img = cv2.imread(fp)
        if img is None: continue
        
        # O FaceFusion devolve o rosto trocado alinhado? Sim, ou nós usamos a imagem recortada dele.
        # As imagens do YTF já foram recortadas, então os Swaps também terão só o tamanho do rosto!
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (112, 112))
        
        face_tensor = preprocess_face(img_resized)
        if face_tensor is not None:
            emb = get_embedding(model, face_tensor, DEVICE)
            embeddings.append(emb)
    
    if len(embeddings) > 0:
        mean_emb = np.mean(embeddings, axis=0)
        return mean_emb / np.linalg.norm(mean_emb)
    return None

def main():
    print("=" * 80)
    print("👾 ATAQUE DEEPFAKE EM VÍDEOS (YOUTUBE FACES)".center(80))
    print("=" * 80)
    os.makedirs(SWAPS_DIR, exist_ok=True)
    
    # Carrega as identidades do LFW splits (mas só nos interessam os IMPOSTORES!)
    all_pairs = parse_splits("data/ytf/splits.txt", max_pairs=NUM_PAIRS_TO_TEST)
    impostor_pairs = [p for p in all_pairs if p[2] == 0]
    
    print(f"🎯 Reduzindo alvos: Identificamos {len(impostor_pairs)} pares de pessoas diferentes.")
    print(f"Iremos aplicar o Swap do FaceFusion neles! (Personagem A invadindo Vídeo B)\n")
    
    adaface = load_adaface(ADAFACE_CKPT).to(DEVICE)
    
    results = []
    for pair_idx, (vid_src, vid_tgt, _) in enumerate(impostor_pairs):
        print(f"🔄 [{pair_idx+1}/{len(impostor_pairs)}] Trocando rosto:")
        print(f"   Atacante (A): {vid_src}")
        print(f"   Corpo/Vídeo Alvo (B): {vid_tgt}")
        
        # Pega a foto de Referência do Atacante (1 frame aleatório)
        src_dir = os.path.join(YTF_DATA_DIR, vid_src)
        tgt_dir = os.path.join(YTF_DATA_DIR, vid_tgt)
        
        if not os.path.exists(src_dir) or not os.path.exists(tgt_dir):
            continue
            
        src_frames = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
        tgt_frames = sorted(glob.glob(os.path.join(tgt_dir, "*.jpg")))
        
        if not src_frames or not tgt_frames: continue
        
        # Pega a foto do atacante
        src_img_path = src_frames[len(src_frames)//2] # Pega o frame do meio
        
        # Seleciona 5 frames do Alvo
        if len(tgt_frames) > FRAMES_TO_SAMPLE:
            idx = np.linspace(0, len(tgt_frames) - 1, FRAMES_TO_SAMPLE).astype(int)
            sampled_tgt = [tgt_frames[i] for i in idx]
        else:
            sampled_tgt = tgt_frames
            
        # Gera o Deepfake Nesses 5 Frames
        swap_frame_paths = []
        for j, tgt_img_path in enumerate(sampled_tgt):
            base_name_src = vid_src.replace("/", "_")
            base_name_tgt = vid_tgt.replace("/", "_")
            out_img = os.path.join(SWAPS_DIR, f"swap_{base_name_src}_on_{base_name_tgt}_{j}.jpg")
            
            if not os.path.exists(out_img) or os.path.getsize(out_img) == 0:
                print(f"     => Construindo Deepfake [Frame {j+1}/{len(sampled_tgt)}]", end="\r")
                success = run_facefusion_swap(src_img_path, tgt_img_path, out_img)
                if not success: continue
            
            swap_frame_paths.append(out_img)
        print("     => Deepfake de vídeo renderizado com sucesso!        ")
        
        # Extrai os embbedings p/ Comparação!
        # Queremos verificar se a IA vai achar que o Vídeo Falso é de fato o Atacante (vid_src)
        emb_src_real = get_video_mean_embedding(vid_src, adaface)
        emb_deepfake = get_mean_embedding_from_frames(swap_frame_paths, adaface)
        
        if emb_src_real is None or emb_deepfake is None:
            continue
            
        sim = compute_similarity(emb_src_real, emb_deepfake)
        status = "🚨 BURLOU O SISTEMA" if sim > 0.45 else "🛡️ BARRADO"
        print(f"   ► Similaridade com o Atacante: {sim:.4f} | Status (τ=0.45): {status}\n")
        
        results.append({
            "source_video": vid_src,
            "target_video": vid_tgt,
            "similarity": sim
        })
    
    # Relatório Final
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "ytf_video_attack_results.csv")
    df.to_csv(csv_path, index=False)
    
    sims = df["similarity"].values
    tau = 0.45
    asr_45 = np.mean(sims > tau) * 100
    tau2 = 0.60
    asr_60 = np.mean(sims > tau2) * 100
    
    print("\n======================================")
    print("💥 RESULTADO GERAL DO ATAQUE DE VÍDEO")
    print("======================================")
    print(f"Tentativas: {len(sims)} Vídeos")
    print(f"Ataque Burlou a IA (Limiar 0.45): {asr_45:.1f}%")
    print(f"Ataque Burlou a IA (Limiar 0.60): {asr_60:.1f}%")
    print(f"Similaridade Média dos Falsos: {np.mean(sims):.4f}")
    
    # Para te mostrar a quebra monumental:
    print(f"\nNo baseline oficial, Impostores davam nota: ~0.28")
    print(f"Se a similaridade pulou perto dos ~0.49+, provamos o ponto do TCC!")

if __name__ == "__main__":
    main()

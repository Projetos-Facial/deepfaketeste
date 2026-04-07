import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Configurações de caminhos
LFW_DIR = "data/lfw/lfw_home/lfw_funneled"
YOLO_FACE_PATH = "data/models/yolov8n-face.pt"
OUTPUT_DIR = "app/results/visualizations"

# Cria a pasta de resultados visuais
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pontos de referência padrão (112x112)
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963], # Olho esquerdo
    [73.5318, 51.5014], # Olho direito
    [56.0252, 71.7366], # Nariz
    [41.5493, 92.3655], # Canto esq boca
    [70.7299, 92.2041]  # Canto dir boca
], dtype=np.float32)

print("Carregando modelo YOLOv8-face...")
yolo = YOLO(YOLO_FACE_PATH)

# Pegar 3 imagens aleatórias (ou as primeiras que encontrar) do LFW
subdirs = [d for d in os.listdir(LFW_DIR) if os.path.isdir(os.path.join(LFW_DIR, d))]
random.seed(42)  # para manter reproduzível
sample_dirs = random.sample(subdirs, 3)

images_to_test = []
for d in sample_dirs:
    person_dir = os.path.join(LFW_DIR, d)
    images = [img for img in os.listdir(person_dir) if img.endswith(".jpg")]
    if images:
        images_to_test.append(os.path.join(person_dir, images[0]))

print(f"Iremos visualizar as imagens: {images_to_test}")

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
fig.suptitle("Detecção, Landmarks e Alinhamento Facial", fontsize=16)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # cores p/ os 5 pontos

for idx, img_path in enumerate(images_to_test):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_draw = img_rgb.copy() # Imagem onde vamos desenhar
    
    results = yolo(img_bgr, verbose=False)
    
    best_conf = 0.0
    best_keypoints = None
    best_box = None

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for i, conf in enumerate(result.boxes.conf):
                if conf.item() > best_conf:
                    best_conf = conf.item()
                    best_box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    if result.keypoints is not None:
                        pts = result.keypoints.xy[i].cpu().numpy()
                        if len(pts) >= 5:
                            best_keypoints = pts[:5]
                            
    aligned_face = None

    if best_box is not None:
        # 1. Desenhar a Bounding Box
        x1, y1, x2, y2 = best_box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2. Desenhar os Landmarks
        if best_keypoints is not None:
            for pt_idx, pt in enumerate(best_keypoints):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(img_draw, (px, py), 4, colors[pt_idx], -1)
            
            # 3. Fazer o Alinhamento Afim!
            tform, _ = cv2.estimateAffinePartial2D(best_keypoints, REFERENCE_FACIAL_POINTS, method=cv2.LMEDS)
            if tform is not None:
                aligned_bgr = cv2.warpAffine(img_bgr, tform, (112, 112), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                aligned_face = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                
    # Plotar Original com Detecções
    ax_orig = axes[idx, 0]
    ax_orig.imshow(img_draw)
    ax_orig.set_title(f"Original + Detecção (Bbox e 5 Pontos)")
    ax_orig.axis("off")
    
    # Plotar o resultado do Alinhamento Facial
    ax_align = axes[idx, 1]
    if aligned_face is not None:
        ax_align.imshow(aligned_face)
        ax_align.set_title("Rosto 112x112 Perfeitamente Alinhado")
    else:
        ax_align.text(0.5, 0.5, "Falhou no alinhamento", ha="center")
    ax_align.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path = os.path.join(OUTPUT_DIR, "exemplo_alinhamento.png")
plt.savefig(save_path, dpi=200)
print(f"Salvo uma imagem comparativa sensacional em: {save_path}")

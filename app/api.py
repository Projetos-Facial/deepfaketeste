import os
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File

# Importamos a lógica e modelos base diretamente do script já feito pra LFW
from baseline_lfw import ADAFACE_CKPT, YOLO_FACE_PATH, FACE_SIZE, DEVICE, load_adaface, detect_and_crop_face, preprocess_face, get_embedding
from ultralytics import YOLO

app = FastAPI(title="Face Verification API (AdaFace+YOLOv8)", version="1.0")

# O carregamento do modelo no nível global garante que eles são instanciados
# apenas uma vez quando a API sobe.
print("🔧 Carregando modelos...")
yolo_model = YOLO(YOLO_FACE_PATH)
adaface_model = load_adaface(ADAFACE_CKPT).to(DEVICE)
print("✅ Modelos carregados!")

@app.post("/verify")
async def verify_faces(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    try:
        contents1 = await img1.read()
        contents2 = await img2.read()
        
        # Converter bytes para imagem np array do OpenCV
        nparr1 = np.frombuffer(contents1, np.uint8)
        nparr2 = np.frombuffer(contents2, np.uint8)
        img_cv1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        img_cv2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

        # Recorte da face com YOLO
        face1 = detect_and_crop_face(yolo_model, img_cv1, target_size=FACE_SIZE)
        face2 = detect_and_crop_face(yolo_model, img_cv2, target_size=FACE_SIZE)

        # Extração das Embeddings
        emb1 = get_embedding(adaface_model, preprocess_face(face1), DEVICE)
        emb2 = get_embedding(adaface_model, preprocess_face(face2), DEVICE)

        # Calcula a Similaridade do Cosseno
        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        
        # Threshold do nosso LFW
        threshold = 0.45
        match = bool(sim >= threshold)
        
        return {
            "match": match,
            "score_cosine": sim,
            "score_percent": f"{sim * 100:.2f}%",
            "threshold": threshold,
            "threshold_percent": f"{threshold * 100:.1f}%"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Roda nossa API na porta 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

# Verificação Facial no LFW — YOLOv8 + AdaFace

Pipeline de **verificação facial** no dataset LFW usando:
- **YOLOv8-Face** — detecção de faces + extração de 5 landmarks
- **AdaFace IR-50** — extração de embeddings faciais (512-d)
- **Similaridade Cosseno** — métrica de verificação

## Requisitos

- Docker Desktop instalado
- GPU NVIDIA + drivers CUDA
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Como Rodar

```bash
# 1. Clonar o repositório
git clone https://github.com/SEU_USUARIO/deepfaketeste.git
cd deepfaketeste

# 2. Subir o container com GPU
docker compose -f docker-compose.gpu.yml up --build -d

# 3. Entrar no container
docker exec -it ia_gpu bash

# 4. (Dentro do container) Testar a GPU
python app/test_gpu.py

# 5. (Dentro do container) Rodar o baseline completo
python app/baseline_lfw.py

# 6. (Dentro do container) Visualizar detecções em 3 imagens de teste
python app/visualize_alignment.py
```

> ⚠️ Na primeira execução, o script baixa automaticamente:
> - Dataset LFW (~233MB)
> - Modelo AdaFace IR-50 (~597MB, via Google Drive)
> - Modelo YOLOv8n-face (~6MB)

## Estrutura do Projeto

```
deepfaketeste/
├── Dockerfile                    # Imagem Docker (CUDA 12.2 + Python 3.10)
├── docker-compose.gpu.yml        # Compose com suporte a GPU
├── docker-compose.cpu.yml        # Compose sem GPU (mais lento)
├── requirements.txt              # Dependências Python
├── pipeline_lfw_adaface.ipynb    # Notebook com passo a passo documentado
│
├── app/
│   ├── baseline_lfw.py           # Script principal do baseline
│   ├── net.py                    # Arquitetura AdaFace (Backbone IR-50)
│   ├── download_lfw.py           # Download auxiliar do LFW
│   ├── test_gpu.py               # Teste rápido de GPU
│   └── visualize_alignment.py    # Gera imagens com detecções visuais
│
└── data/                         # (gerado automaticamente na 1ª execução)
    ├── lfw/                      # Dataset LFW
    └── models/                   # Pesos dos modelos
```

## Resultados Gerados

Após rodar `baseline_lfw.py`, a pasta `app/results/` conterá:

| Arquivo | Descrição |
|---------|-----------|
| `lfw_scores.csv` | Score de similaridade cosseno para cada par |
| `score_hist_baseline.png` | Histograma: genuínos vs. impostores |
| `roc_curve_lfw.png` | Curva ROC com AUC e EER |
| `experimental_config.txt` | Configuração experimental + FAR/FRR |
| `baseline_conclusion.txt` | Conclusão escrita do baseline |

## Pipeline

```
Imagem → YOLOv8-Face → Landmarks (5 pontos) → Alinhamento Afim → AdaFace → Embedding 512-d
                                                                              ↓
Foto A ──────────────────────────────────────────────────────── Embedding A ──┐
                                                                              ├→ Cosine Similarity → Decisão
Foto B ──────────────────────────────────────────────────────── Embedding B ──┘
```

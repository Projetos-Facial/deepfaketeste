# Verificação Facial e Avaliação de Robustez contra Deepfakes

Pipeline completo de **verificação facial** e **avaliação de vulnerabilidade a ataques de face swap (deepfakes)** utilizando:

- **YOLOv8-Face** — detecção de faces + extração de 5 landmarks para alinhamento
- **AdaFace IR-50** — extração de embeddings faciais (512-d) pré-treinado no WebFace4M
- **FaceFusion (inswapper_128)** — geração automatizada de face swaps (deepfakes)
- **Similaridade Cosseno** — métrica de verificação biométrica

## Objetivo

Avaliar a **robustez do sistema biométrico** (YOLOv8 + AdaFace) contra ataques de **deepfake por face swap**, medindo a taxa com que imagens manipuladas conseguem enganar o classificador facial. Os experimentos são conduzidos em dois cenários:

1. **Dataset Pessoal** — imagens controladas de pessoas conhecidas
2. **LFW (Labeled Faces in the Wild)** — benchmark acadêmico padrão com 13.233 imagens

---

## Requisitos

- Docker instalado (versão com suporte a `docker-compose`)
- GPU NVIDIA + drivers CUDA
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Como Inicializar o Ambiente (Docker)

Todo o projeto roda dentro de um container Docker configurado com a biblioteca CUDA, garantindo acesso direto à sua placa de vídeo.

```bash
# 1. No terminal do seu servidor/computador, suba o container:
sudo docker-compose -f docker-compose.gpu.yml up -d

# 2. Agora "entre" no terminal do container para poder executar os Python scripts:
sudo docker exec -it ia_gpu bash

# 🪄 Pronto! Seu terminal vai mudar para algo parecido com "root@a2bd43...:/app#". 
# Todos os comandos "python app/nome_do_arquivo.py" a seguir devem ser digitados AQUI dentro!
```

> **Dica:** Se aparecer um erro no passo 1 dizendo que "o container ia_gpu já existe", force a remoção com `sudo docker rm -f ia_gpu` e tente de novo.

---

## Estrutura do Projeto & Como Rodar Cada Script

Para evitar confusão, os scripts estão organizados em **Núcleo Acadêmico (Paper)** e **Utilitários Extras**.

### 📚 1. Núcleo Acadêmico do PAPER (pasta `app/`)
Estes são os arquivos centrais responsáveis por gerar os dados da sua pesquisa LFW.

| O que ele faz? | Como rodar no terminal do Docker? |
|---|---|
| **`app/baseline_lfw.py`**<br>Avalia 6000 pares reais do protocolo LFW. Extrai a acurácia, FAR, e FRR "originais" da IA para usarmos de base. É a métrica do mundo honesto. | `python app/baseline_lfw.py` |
| **`app/run_swaps_lfw.py`**<br>**(O Ataque Principal do Artigo)**. Gera automaticamente 100 deepfakes usando pessoas do LFW, injeta contra a IA, calcula Testes Estatísticos (IC 95%) e salva os gráficos. | `python app/run_swaps_lfw.py` |
| **`app/generate_paper_tables.py`**<br>Junta tudo que foi gerado pelos scripts acima e constrói as tabelas formatadas de Resumo e Consolidação que vão direto pro Word do artigo. | `python app/generate_paper_tables.py` |
| **`app/api.py`**<br>Transforma o sistema inteiro num serviço de internet "FastAPI". Se você quiser conectar essa biometria em um aplicativo real futuramente, é ele a porta de entrada. | `uvicorn app.api:app --host 0.0.0.0 --port 8000` |
| **`app/net.py`**<br>É onde o cérebro matemático "Backbone IResNet-50" (do AdaFace) fica instalado. Você não roda esse arquivo, os outros scripts o chamam. | *(Apenas importado pelos outros)* |

<br>

### 🛠️ 2. Utilitários Livres & Testes com Amigos (pasta `app/utils/`)
Esses arquivos fazem a mesmíssima coisa, só que aplicados nas fotos soltas da sua equipe e dataset pessoal, ótimos para tirar "prova real" validada.

| O que ele faz? | Como rodar no terminal do Docker? |
|---|---|
| **`app/run_swaps_experiment.py`**<br>Cruza as pessoas do seu arquivo `data/dataset_pessoal/originais`, arranca o rosto de um, cola no corpo do outro e joga contra a IA. | `python app/run_swaps_experiment.py` |
| **`app/utils/compare_faces.py`**<br>Pega duas imagens quaisquer e diz se a IA acha a pessoa igual ou não. | `python app/utils/compare_faces.py data/rostoA.png data/rostoB.png` |
| **`app/utils/swap_faces.py`**<br>Faz manualmente 1 único deepfake e salva onde você mandar. Bom para criar imagens engraçadas fora da base de testes. | `python app/utils/swap_faces.py data/rosto.jpg data/corpo_alvo.png resultado.jpg` |
| **`app/utils/run_experiment.py`**<br>Lê toooodas as fotos reais da sua equipe e compara "todo mundo contra todo mundo" e gera uma tabela Top 15. | `python app/utils/run_experiment.py` |
| **`app/utils/visualize_alignment.py`**<br>Um utilitário visual que gera uma imagem em PDF demonstrando os 5 pontinhos verdes (Nariz, Boca e Olho) que a YOLO enxerga. Excelente ferramenta pra colocar no artigo como contexto teórico! | `python app/utils/visualize_alignment.py` |
| **`app/utils/test_gpu.py`**<br>Script rápido pra validar se o Docker de fato está sugando as informações da sua placa de vídeo NVIDIA. | `python app/utils/test_gpu.py` |

---

## Pipeline de Avaliação

```
                    ┌─────────────────────────────────────────┐
                    │         ETAPA 1: GERAÇÃO DE FRAUDE      │
                    │                                         │
  Foto Rosto A ───► │   FaceFusion (inswapper_128)            │ ──► Imagem Swap
  Foto Corpo B ───► │   Cola rosto A no corpo de B            │     (Deepfake)
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │       ETAPA 2: AUDITORIA BIOMÉTRICA     │
                    │                                         │
   Imagem Swap ───► │  YOLOv8-Face → Crop → AdaFace → Emb.   │ ──► Similaridade
  Foto Original ──► │  YOLOv8-Face → Crop → AdaFace → Emb.   │     Cosseno
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │         ETAPA 3: VEREDICTO              │
                    │                                         │
                    │  sim >= τ  →  🚨 Deepfake ENGANOU a IA  │
                    │  sim <  τ  →  🛡️ Deepfake foi BARRADO   │
                    └─────────────────────────────────────────┘
```

---

## Resultados

### 1. Baseline — LFW (sem manipulação)

| Métrica | Valor |
|---------|-------|
| Pares avaliados | 6.000 (3.000 genuínos + 3.000 impostores) |
| Acurácia | **95.97%** (threshold ótimo = 0.238) |
| AUC | **0.9660** |
| EER | **6.47%** |
| FAR (τ=0.45) | 0.00% |
| FRR (τ=0.45) | 12.40% |
| FAR (τ=0.60) | 0.00% |
| FRR (τ=0.60) | 38.13% |

### 2. Ataque Deepfake — Dataset Pessoal

| Swap | Alvo | Similaridade | τ=0.45 |
|------|------|:------------:|--------|
| Paulo → João | Paulo | Alto | 🚨 Enganou |
| Lucas → Carol | Lucas | Alto | 🚨 Enganou |
| Carol → Paulo | Carol | Alto | 🚨 Enganou |

### 3. Ataque Deepfake — LFW (30 swaps)

| Métrica | τ = 0.45 | τ = 0.60 |
|---------|:--------:|:--------:|
| **Attack Success Rate (ASR)** | **93.3%** (28/30) | **90.0%** (27/30) |
| Swaps barrados | 6.7% (2/30) | 10.0% (3/30) |

| Estatística | Valor |
|-------------|-------|
| Similaridade Média | 0.6846 |
| Similaridade Mediana | 0.7565 |
| Mín | -0.1167 |
| Máx | 0.8232 |

### Interpretação

> O sistema biométrico baseado em AdaFace IR-50 apresenta **alta vulnerabilidade**
> a ataques de face swap (deepfakes): **93.3%** dos swaps enganaram a IA com τ=0.45.
> Os deepfakes gerados pelo FaceFusion (inswapper_128) produzem similaridades na
> faixa de 0.60–0.82, equivalente à faixa de **pares genuínos** do baseline.
> Isso demonstra a necessidade de mecanismos complementares como **Liveness Detection**.

---

## Arquivos de Resultados

| Arquivo | Descrição |
|---------|-----------|
| `app/results/lfw_scores.csv` | Scores cosseno dos 6.000 pares do baseline LFW |
| `app/results/lfw_swap_attack_results.csv` | Detalhes de cada swap no LFW (30 ataques) |
| `app/results/score_hist_baseline.png` | Histograma: genuínos vs. impostores (baseline) |
| `app/results/roc_curve_lfw.png` | Curva ROC com AUC e EER |
| `app/results/lfw_swap_attack_histogram.png` | Distribuição dos scores dos deepfakes |
| `app/results/lfw_baseline_vs_attack.png` | Comparação baseline vs. ataque deepfake |
| `app/results/experimental_config.txt` | Configuração experimental completa |
| `app/results/baseline_conclusion.txt` | Conclusão do baseline |
| `app/results/lfw_swap_attack_conclusion.txt` | Conclusão do ataque deepfake |
| `data/dataset_pessoal/resultados_dataset_pessoal.csv` | Comparação cruzada do dataset pessoal |

---

## Tecnologias

| Componente | Tecnologia | Versão |
|-----------|-----------|--------|
| Container | Docker + NVIDIA CUDA | 12.2.2 + cuDNN 8 |
| Linguagem | Python | 3.10 |
| Deep Learning | PyTorch | 2.11+ |
| Detecção Facial | YOLOv8n-face (Ultralytics) | 8.4+ |
| Embedding Facial | AdaFace IR-50 (WebFace4M) | — |
| Face Swap | FaceFusion (inswapper_128) | 3.x |
| API | FastAPI + Uvicorn | — |

---

## Referências

- **AdaFace**: Kim, M. et al. "AdaFace: Quality Adaptive Margin for Face Recognition." CVPR, 2022.
- **LFW**: Huang, G. B. et al. "Labeled Faces in the Wild." University of Massachusetts, 2007.
- **FaceFusion**: https://github.com/facefusion/facefusion
- **YOLOv8-Face**: https://github.com/akanametov/yolo-face

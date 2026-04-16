FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Instalar Python 3.10 e TODAS as dependências do sistema
# (inclui curl e ffmpeg que o FaceFusion exige)
RUN apt update && apt install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Setar python padrão
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Instalar dependências Python do projeto
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Instalar onnxruntime-gpu (necessário para o FaceFusion)
RUN pip install onnxruntime-gpu

# Instalar dependências do FaceFusion
COPY facefusion/requirements.txt /tmp/facefusion_requirements.txt
RUN pip install -r /tmp/facefusion_requirements.txt && rm /tmp/facefusion_requirements.txt

COPY . .

CMD ["bash"]

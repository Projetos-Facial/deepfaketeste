FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Instalar Python 3.10 e dependências do sistema
RUN apt update && apt install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setar python padrão
RUN ln -s /usr/bin/python3.10 /usr/bin/python

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]

# Use CUDA base image — GPU detection + nvidia-smi work out of the box.
# Still runs on CPU when --gpus flag is omitted in docker-compose.
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.10 + system libs
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libpcap-dev \
    git \
    curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && curl -fsSL https://tailscale.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN sed -i '/pywin32/d' requirements.txt && \
    sed -i '/torch-directml/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50051 5678 52415

RUN mkdir -p models

CMD ["python", "main.py", "run", "--default-model", "qwen2.5:0.5b"]

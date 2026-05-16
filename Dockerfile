# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - build-essential: for compiling some python packages
# - libgl1-mesa-glx & libglib2.0-0: required for OpenCV
# - libpcap-dev: required for Scapy
# - git: to clone/download models if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libpcap-dev \
    git \
    curl \
    && curl -fsSL https://tailscale.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Remove Windows-specific dependencies and DirectML (which is for Windows)
# Use standard torch for Linux
RUN sed -i '/pywin32/d' requirements.txt && \
    sed -i '/torch-directml/d' requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose ports:
# 50051: Node Port (P2P/GRPC)
# 5678: Discovery Port (UDP)
# 52415: Web UI / ChatGPT API Port
EXPOSE 50051 5678 52415

# Create models directory to ensure it exists
RUN mkdir -p models

# The command to run the application
# We use "run" command by default
CMD ["python", "main.py", "run"]

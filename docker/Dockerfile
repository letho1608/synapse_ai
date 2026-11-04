# Synapse node on Linux (CPU torch). Used by docker/docker-compose.yml to run
# a multi-node cluster on one machine, each node with its own network stack.
FROM python:3.10-slim

# iproute2: `tc netem` for simulating latency / loss between nodes.
RUN apt-get update \
 && apt-get install -y --no-install-recommends iproute2 iputils-ping \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements-base.txt backend/requirements-base.txt
# CPU torch from the PyTorch index, everything else from PyPI (the PyTorch
# index's own copies of some deps have metadata older pips reject).
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
 && pip install --no-cache-dir -r backend/requirements-base.txt

COPY main.py ./
COPY backend backend
# Built web UI if present (npm --prefix frontend run build); the node serves a
# basic status page without it.
COPY frontend frontend

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
EXPOSE 46051 47051 52415 5678/udp
ENTRYPOINT ["python", "main.py"]

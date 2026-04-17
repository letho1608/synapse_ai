# Synapse AI: Decentralized Distributed AI Supercomputing Cluster

**Distributed AI Supercluster** — Run LLMs across multiple nodes, OpenAI-compatible API, unified model management & training from one single dashboard.

[![Repo](https://img.shields.io/badge/repo-github.com%2Fletho1608%2Fsynapse__ai-blue)](https://github.com/letho1608/synapse_ai)

---

## Overview

Synapse AI Server is a powerful **decentralized distributed supercomputing system** that allows you to run large language models (LLMs) across multiple machines over Tailscale. Each node in the cluster handles a portion of the model (layer sharding) and communicates via gRPC. You get **one unified web UI** to manage models, chat, train, and monitor the entire cluster status without requiring a single high-end machine.

> **Important:** When using this repo with Tailscale discovery, **you must install and run the Tailscale app** on your machine (the Tailscale app must be running — [tailscale.com/download](https://tailscale.com/download)). If Tailscale is not running, cluster discovery and the Monitoring page will not work correctly.

---

## Features

| Feature | Description |
|---------|-------------|
| **Distributed inference** | Run LLMs (PyTorch + Hugging Face) across multiple nodes with layer sharding over Tailscale. |
| **OpenAI API** | `/v1/chat/completions` endpoint (streaming / non-streaming), OpenAI-compatible. |
| **500+ Hugging Face models** | Automatic model discovery and management; pull/delete directly from Web UI. |
| **Web UI** | Dashboard, Model management, Chat with model, Training (LoRA), Dataset, Monitoring (Tailscale nodes), Settings, About. |
| **LoRA training** | Fine-tune from Web UI; pipeline load-sharing when multiple nodes are available. |
| **Cluster monitoring** | View Tailscale nodes, total CPU/RAM/GPU/disk, and per-node status. |

---

## Requirements

- **Python** 3.10+
- **Tailscale** — install [Tailscale](https://tailscale.com/download) and **keep the Tailscale app running** when using this repo (cluster discovery & Monitoring depend on it).
- **NVIDIA GPU** (optional; runs on CPU but slower)
- **Windows** (code currently uses `windows_device_capabilities`; Linux may need additions)

---

## Installation

```bash
git clone https://github.com/letho1608/synapse_ai.git
cd synapse_ai
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Run

1. **Start the Tailscale app** on your machine (ensure Tailscale is running).
2. Configure API key & tailnet (see [Tailscale configuration](#tailscale-configuration)).
3. Run:

```bash
python main.py
```

- **Entry point:** Root `main.py` only calls `synapse.main`; all logic and config live in `synapse/main.py`.
- After starting, open the URL printed in the terminal (e.g. `http://127.0.0.1:52415`).
- **Web UI:** Dashboard, Model management, Chat, Training, Dataset, Monitoring, Settings, About.
- **API:** `POST http://<host>:52415/v1/chat/completions` with a body like [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat).

---

## Tailscale configuration

**Location:** Tailscale config is in **`synapse/main.py`**. The repo does not contain real API key or tailnet (only placeholders). You supply key/tailnet via **environment variables** (`.env`) or **command-line arguments**.

> Remember to **run the Tailscale app** before running `python main.py` when using Tailscale discovery.

### Option 1: `.env` file (recommended)

1. Copy `.env.example` → `.env` in the repo root.
2. Edit `.env` and set `TAILSCALE_API_KEY` and `TAILNET_NAME`.
3. `.env` is loaded automatically when you run `python main.py` (and is in `.gitignore`).

```bash
cp .env.example .env
# Edit .env: TAILSCALE_API_KEY=tskey-api-xxx, TAILNET_NAME=your-tailnet
python main.py
```

- **API key:** [Tailscale Admin](https://login.tailscale.com/admin/settings/keys) → Generate API key.
- **Tailnet name:** Your tailnet name (e.g. `your-org.github` or email depending on signup).

### Option 2: Command-line arguments

```bash
python main.py run --tailscale-api-key "tskey-api-xxx" --tailnet-name "your-tailnet"
```

**Note:** Do not commit `.env` or real keys to GitHub.

---

## Other configuration

| Item | Location |
|------|----------|
| **Default model** | Web UI → Settings, or file `synapse/config/settings.json` (`default_model`). |
| **API port** | Default `52415`; override with `--chatgpt-api-port` (e.g. `python main.py run ...`). |
| **Discovery** | Default `tailscale`; can use `udp` or `manual` with `--discovery-module` and `--discovery-config-path`. |

---

## Project structure (summary)

```
synapse_ai/
├── main.py                 # Entry point (calls synapse.main)
├── requirements.txt
├── README.md
├── synapse/
│   ├── main.py             # Main logic; reads Tailscale from env or CLI
│   ├── api/                # ChatGPT API, routes
│   ├── inference/          # PyTorch/HF engine, shard
│   ├── networking/        # Tailscale, gRPC, UDP, manual discovery
│   ├── topology/           # Device capabilities, partitioning
│   ├── training/           # LoRA, dataset
│   ├── config/             # settings.json
│   ├── data/               # Datasets (json/jsonl)
│   ├── tinychat/           # Web UI (dashboard.html, production.css)
│   └── model_list.py       # 500+ Hugging Face models (Auto-discovery)
└── tests/
```

---

## Main API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/distributed/status` | Distributed status (partitions, multi-node). |
| GET | `/v1/tailscale/nodes` | Tailscale nodes list (for Web Monitoring). |
| GET | `/v1/models` | Model list (downloaded + available to pull). |
| POST | `/v1/chat/completions` | Chat completion (OpenAI-compatible). |
| POST | `/v1/models/pull` | Pull model (body: `model`). |
| POST | `/v1/training/start` | Start training (body: model, dataset, epochs, ...). |

---

## Tests

```bash
pytest tests/ -v
```

---

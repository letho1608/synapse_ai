#!/bin/sh
# Synapse AI installer (Linux/macOS). Thin wrapper, no new runtime.
# Installs the default node closure (base; Windows-only lines are skipped by markers).
set -eu
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[FAIL] python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi
echo "[INFO] $(python3 --version)"

if [ ! -x ".venv/bin/python" ]; then
  echo "[INFO] creating .venv ..."
  python3 -m venv .venv
fi

echo "[INFO] pip install -r backend/requirements.txt ..."
.venv/bin/python -m pip install -r backend/requirements.txt

echo "[INFO] torch backend detect:"
.venv/bin/python -c "import torch; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

if command -v npm >/dev/null 2>&1; then
  echo "[INFO] building the web UI ..."
  if npm --prefix frontend install --no-audit --no-fund; then
    npm --prefix frontend run build || echo "[WARN] web UI build failed" >&2
  else
    echo "[WARN] npm install failed: web UI not built" >&2
  fi
else
  echo "[WARN] Node.js/npm not found: the web UI is not built (the node serves a basic status page)." >&2
  echo "       Install Node.js 18+, then: npm --prefix frontend install && npm --prefix frontend run build" >&2
fi

cat <<'EOF'

[OK] installed. Next steps:
  source .venv/bin/activate
  python main.py            # then open http://127.0.0.1:52415

Optional extras:
  .venv/bin/python -m pip install -r backend/requirements-dev.txt     # packaging/dev
EOF

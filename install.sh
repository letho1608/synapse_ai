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

# frontend/dist ships prebuilt in git; rebuilding here would dirty the checkout.
if [ -f frontend/dist/index.html ]; then
  echo "[INFO] web UI: prebuilt frontend/dist (rebuild after editing frontend/src: npm --prefix frontend run build)"
elif command -v npm >/dev/null 2>&1; then
  echo "[INFO] building the web UI ..."
  if npm --prefix frontend install --no-audit --no-fund; then
    npm --prefix frontend run build || echo "[WARN] web UI build failed: keeping the prebuilt one" >&2
  else
    echo "[WARN] npm install failed: keeping the prebuilt web UI" >&2
  fi
else
  echo "[INFO] Node.js/npm not found: using the prebuilt web UI in frontend/dist."
  echo "       To rebuild after editing frontend/src: npm --prefix frontend install && npm --prefix frontend run build"
fi

cat <<'EOF'

[OK] installed. Next steps:
  source .venv/bin/activate
  python main.py            # then open http://127.0.0.1:52415

Optional extras:
  .venv/bin/python -m pip install -r backend/requirements-dev.txt     # packaging/dev
EOF

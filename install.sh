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

cat <<'EOF'

[OK] installed. Next steps:
  source .venv/bin/activate
  python main.py

Optional extras:
  .venv/bin/python -m pip install -r backend/requirements-dev.txt     # packaging/dev
EOF

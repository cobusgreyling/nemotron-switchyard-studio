#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Prefer a Python that already has torch/peft/fastapi (this Mac does).
if python3 -c "import torch, fastapi, peft, transformers, openai, dotenv, uvicorn" 2>/dev/null; then
  PYTHON=python3
else
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
  else
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
  PYTHON=python
fi

if [[ ! -f .env ]]; then
  echo "No .env found — copy .env.example and set NVIDIA_API_KEY for live NIM."
fi

exec "$PYTHON" app.py

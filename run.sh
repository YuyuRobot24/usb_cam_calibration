#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing virtual environment: .venv"
  echo "Create it with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install --upgrade pip setuptools wheel"
  echo "  python -m pip install -r requirements.txt"
  exit 1
fi

exec .venv/bin/python app.py

#!/usr/bin/env bash
# One-time environment setup for the minimal DP image generation pipeline.
#
#   - installs Python requirements
#   - clones the efficientvit repo (DC-AE model definition; not on PyPI)
#   - downloads the DC-AE weights from HuggingFace (cached)
#
# Re-running is safe: each step is skipped if already done.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Python deps
if [ "${SKIP_PIP:-0}" != "1" ]; then
  echo "[setup] Installing Python requirements"
  pip install -q -r requirements.txt
fi

# 2. efficientvit (DC-AE) — vendored alongside this repo by default.
#    Override the location with EFFICIENTVIT_DIR=/some/path ./setup.sh
EFFICIENTVIT_DIR="${EFFICIENTVIT_DIR:-$SCRIPT_DIR/efficientvit}"
if [ ! -d "$EFFICIENTVIT_DIR/efficientvit" ]; then
  echo "[setup] Cloning efficientvit -> $EFFICIENTVIT_DIR"
  git clone --depth 1 https://github.com/mit-han-lab/efficientvit "$EFFICIENTVIT_DIR"
else
  echo "[setup] efficientvit already present at $EFFICIENTVIT_DIR"
fi

# 3. Pre-fetch DC-AE weights so the first run is offline-friendly.
echo "[setup] Downloading DC-AE weights from HuggingFace (cached)"
EFFICIENTVIT_DIR="$EFFICIENTVIT_DIR" python - <<'PY'
from codec import load_dcae
load_dcae(device="cpu")
print("[setup] DC-AE weights ready.")
PY

echo "[setup] Done. Run with: ./run.sh"

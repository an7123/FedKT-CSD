#!/usr/bin/env bash
# Minimal DP synthetic image generation — end-to-end runner.
#
# Reproduces the DC-AE + DP-sufficient-statistics baseline on EuroSAT
# (ε=10, δ=1e-5). Expected accuracy ≈ 82% with 200 epochs.
#
# Run ./setup.sh first to install deps and clone efficientvit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-bootstrap if setup hasn't been run.
if [ ! -d "${EFFICIENTVIT_DIR:-$SCRIPT_DIR/efficientvit}/efficientvit" ] \
   && [ ! -d "/tmp/efficientvit/efficientvit" ]; then
  echo "[run] efficientvit not found — running setup.sh"
  ./setup.sh
fi

DATASET="${DATASET:-eurosat}"
EPSILON="${EPSILON:-10}"
EPOCHS="${EPOCHS:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
DATA_DIR="${DATA_DIR:-./data}"

echo "[run] dataset=$DATASET epsilon=$EPSILON epochs=$EPOCHS"
python main.py \
  --dataset "$DATASET" \
  --data-dir "$DATA_DIR" \
  --epsilon "$EPSILON" \
  --epochs "$EPOCHS" \
  --output-dir "$OUTPUT_DIR"

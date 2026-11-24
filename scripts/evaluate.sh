#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[*] Evaluating HPMA Model on Validation Set..."
python3 eval.py --base_config configs/base_config.yaml --data_config configs/isles2022.yaml \
  --checkpoint checkpoints/hpma_isles-2022_best.pt
echo "[✓] Evaluation completed."

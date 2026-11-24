#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[*] Starting HPMA Model Training..."
python3 train.py --base_config configs/base_config.yaml --data_config configs/isles2022.yaml
echo "[✓] HPMA Training completed successfully."

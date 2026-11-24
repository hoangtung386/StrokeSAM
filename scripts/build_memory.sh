#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[*] Running Offline Prototype Memory Bank Construction..."
python3 build_memory.py --config configs/isles2022.yaml
echo "[✓] Prototype memory extraction finished successfully."

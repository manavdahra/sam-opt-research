#!/usr/bin/env bash
# GPU 2 — Reparametrisation experiments (ResNet-18 + ViT)
# Run as: bash experiments/run_gpu2.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "GPU 2: Reparametrisation experiments"
echo "Started: $(date)"
echo "=========================================="

echo ""
echo "[1/2] ResNet-18 reparam"
uv run experiments/run_reparam.py --config configs/resnet18_reparam.yaml

echo ""
echo "[2/2] ViT reparam"
uv run experiments/run_reparam.py --config configs/vit_reparam.yaml

echo ""
echo "=========================================="
echo "GPU 2 done: $(date)"
echo "=========================================="

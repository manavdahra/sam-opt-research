#!/usr/bin/env bash
# GPU 1 — Baseline experiments (ResNet-18 + ViT)
# Run as: bash experiments/run_gpu1.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "GPU 1: Baseline experiments"
echo "Started: $(date)"
echo "=========================================="

echo ""
echo "[1/2] ResNet-18 baseline"
uv run experiments/run_baseline.py --config configs/resnet18_baseline.yaml

echo ""
echo "[2/2] ViT baseline"
uv run experiments/run_baseline.py --config configs/vit_baseline.yaml

echo ""
echo "=========================================="
echo "GPU 1 done: $(date)"
echo "=========================================="

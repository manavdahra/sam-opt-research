#!/usr/bin/env bash
# ViT reparametrisation experiments
# Run as: bash experiments/run_vit_reparam.sh

set -euo pipefail

echo "=========================================="
echo "ViT reparam"
echo "Started: $(date)"
echo "=========================================="

uv run experiments/run_reparam.py --config configs/vit_reparam_run.yaml

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="

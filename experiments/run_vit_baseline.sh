#!/usr/bin/env bash
# ViT baseline
# Run as: bash experiments/run_vit_baseline.sh

set -euo pipefail

echo "=========================================="
echo "ViT baseline"
echo "Started: $(date)"
echo "=========================================="

uv run experiments/run_baseline.py --config configs/vit_baseline_run.yaml

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="

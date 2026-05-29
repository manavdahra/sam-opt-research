#!/usr/bin/env bash
# ResNet-18 reparametrisation experiments
# Run as: bash experiments/run_resnet18_reparam.sh

set -euo pipefail

echo "=========================================="
echo "ResNet-18 reparam"
echo "Started: $(date)"
echo "=========================================="

uv run experiments/run_reparam.py --config configs/resnet18_reparam_run.yaml

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="

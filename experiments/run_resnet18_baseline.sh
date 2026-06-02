#!/usr/bin/env bash
# ResNet-18 baseline
# Run as: bash experiments/run_resnet18_baseline.sh

set -euo pipefail

echo "=========================================="
echo "ResNet-18 baseline"
echo "Started: $(date)"
echo "=========================================="

uv run experiments/run_baseline.py --config configs/resnet18_baseline_run.yaml

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="

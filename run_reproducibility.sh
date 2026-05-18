#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Hybrid IFCM-TransUNet-FMRCNN Emphysema Reproducibility Run "
echo "============================================================"

echo ""
echo "[1/5] Checking Python environment..."
python --version

echo ""
echo "[2/5] Creating output directories..."
mkdir -p outputs/checkpoints
mkdir -p outputs/figures
mkdir -p outputs/predictions
mkdir -p outputs/metrics

echo ""
echo "[3/5] Starting model training..."
python train.py --config config.yaml

echo ""
echo "[4/5] Running quantitative evaluation..."
python evaluate.py --config config.yaml

echo ""
echo "[5/5] Running inference and generating visual outputs..."
python inference.py --config config.yaml

echo ""
echo "============================================================"
echo " Reproducibility pipeline completed successfully."
echo " Outputs saved in: outputs/"
echo "============================================================"

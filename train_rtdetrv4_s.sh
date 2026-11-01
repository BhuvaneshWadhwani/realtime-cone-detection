#!/bin/bash
#SBATCH --job-name=rtdetrv4_s_fsoco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rtdetrv4_s_%j.out
#SBATCH --error=logs/rtdetrv4_s_%j.err

set -e

PROJECT_DIR="$HOME/Real Time Detection Transformer"
REPO_DIR="$PROJECT_DIR/RT-DETRv4"
CONFIG="configs/rtv4/rtv4_hgnetv2_s_fsoco.yml"

cd "$PROJECT_DIR"
mkdir -p logs

source ~/.bashrc

conda activate rtdetrv4

cd "$REPO_DIR"

echo "Running from: $(pwd)"
echo "Using config: $CONFIG"
echo "Python: $(which python)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

torchrun \
  --nproc_per_node=1 \
  --master_port=7777 \
  train.py \
  -c "$CONFIG" \
  --use-amp \
  --seed=0
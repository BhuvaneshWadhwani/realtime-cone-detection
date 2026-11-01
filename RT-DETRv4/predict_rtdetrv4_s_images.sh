#!/bin/bash
#$ -N predict_rtdetrv4_s_fsoco
#$ -q gpu
#$ -cwd
#$ -j y
#$ -o logs/predict_rtdetrv4_s_$JOB_ID.out
#$ -l h_rt=02:00:00
#$ -l h_vmem=64G
#$ -pe smp 8
#$ -l gpu=1

module load cuda/12.8
module load python/OLD/3.10.4

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/Real Time Detection Transformer/RT-DETRv4/dinov3:$HOME/Real Time Detection Transformer/RT-DETRv4:$PYTHONPATH"

cd "$HOME/Real Time Detection Transformer/RT-DETRv4"

echo "=== RT-DETRv4-S FSOCO IMAGE PREDICTION ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Checkpoint: outputs/rtv4_hgnetv2_s_fsoco/best_stg2.pth"

nvidia-smi

python3 tools/inference/fsoco_predict_images.py \
  -c configs/rtv4/rtv4_hgnetv2_s_fsoco.yml \
  -r outputs/rtv4_hgnetv2_s_fsoco/best_stg2.pth \
  -i ../data/fsoco_coco/images/test \
  -o outputs/rtv4_hgnetv2_s_fsoco/predictions_test_conf040 \
  --imgsz 640 \
  --conf 0.40 \
  --max-images 500 \
  -d cuda

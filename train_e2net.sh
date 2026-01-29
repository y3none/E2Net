#!/bin/bash

# E2Net训练脚本
# 使用DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m) 作为预训练编码器

python train_e2net.py \
    --train_dataset '../dataset/TrainDataset' \
    --val_dataset '../dataset/TestDataset/CAMO' \
    --val_interval 1 \
    \
    --encoder_name 'facebook/dinov3-vitb16-pretrain-lvd1689m' \
    --encoder_pretrained 'dinov3_models/vitb16' \
    --freeze_encoder \
    --feature_dim 256 \
    --use_simple_encoder \
    \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 5e-4 \
    \
    --lambda1 2.5 \
    --lambda2 1.0 \
    --lambda3 0.2 \
    --lambda_edge 0.7 \
    --lambda_iou 1.0 \
    \
    --save_dir 'checkpoint/E2Net_DINOv3' \
    --log_dir 'logs/E2Net_DINOv3' \
    --save_interval 10 \
    \
    --num_workers 4 \
    --device 'cuda'

echo ""
echo "======================================================================"
echo "Training completed!"
echo "======================================================================"
#!/bin/bash

# E2Net测试脚本
# 在多个数据集上测试训练好的E2Net模型

python test_e2net.py \
    --checkpoint 'checkpoint/E2Net_DINOv3/E2Net_best.pth' \
    --encoder_name 'facebook/dinov3-vitb16-pretrain-lvd1689m' \
    --encoder_pretrained 'dinov3_models/vitb16' \
    --feature_dim 256 \
    --use_simple_encoder \
    --test_dataset '../dataset/TestDataset' \
    --output_dir 'output/Prediction/E2Net_DINOv3-test'

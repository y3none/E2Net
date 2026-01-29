#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E2Net训练脚本 - 仅使用改进的损失函数
最小改动版本，只修改损失函数部分
"""

import os
import sys
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from tqdm import tqdm

# 导入数据集
import dataset

# 导入改进的损失函数
from loss import ImprovedLoss

# 导入E2Net
from E2Net import build_e2net


def train_epoch(model, train_loader, optimizer, criterion, epoch, writer, device='cuda'):
    """训练一个epoch"""
    model.train()
    
    # 如果encoder是冻结的，确保它保持eval模式
    if hasattr(model, 'encoder'):
        model.encoder.eval()
    
    # 统计
    total_loss = 0
    loss_components = {
        'dice': 0,
        'bce': 0,
        'iou': 0,
        'edge': 0,
        'aux': 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=120)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        Y_hat, M_coarse = model(images)
        
        # 计算损失
        loss, loss_dict = criterion(Y_hat, M_coarse, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累计损失
        total_loss += loss_dict['total']
        loss_components['dice'] += loss_dict['dice_main']
        loss_components['bce'] += loss_dict['bce_main']
        loss_components['iou'] += loss_dict['iou_main']
        loss_components['edge'] += loss_dict['edge_main']
        loss_components['aux'] += loss_dict['aux']
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss_dict["total"]:.4f}',
            'Dice': f'{loss_dict["dice_main"]:.4f}',
            'IoU': f'{loss_dict["iou_main"]:.4f}'
        })
        
        # # 每50个batch详细输出
        # if batch_idx % 50 == 0:
        #     avg_loss = total_loss / (batch_idx + 1)
        #     print(f'\n  Batch {batch_idx}/{len(train_loader)} | '
        #           f'Loss: {loss_dict["total"]:.4f} | '
        #           f'Dice: {loss_dict["dice_main"]:.4f} | '
        #           f'IoU: {loss_dict["iou_main"]:.4f} | '
        #           f'Edge: {loss_dict["edge_main"]:.4f}')
    
    # 计算平均指标
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches
    
    # TensorBoard记录
    if writer is not None:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/Dice', loss_components['dice'], epoch)
        writer.add_scalar('Train/BCE', loss_components['bce'], epoch)
        writer.add_scalar('Train/IoU', loss_components['iou'], epoch)
        writer.add_scalar('Train/Edge', loss_components['edge'], epoch)
        writer.add_scalar('Train/Aux', loss_components['aux'], epoch)
    
    print(f'\nEpoch {epoch} Summary: '
          f'Loss={avg_loss:.4f}, '
          f'Dice={loss_components["dice"]:.4f}, '
          f'IoU={loss_components["iou"]:.4f}, '
          f'Edge={loss_components["edge"]:.4f}')
    
    return avg_loss


def validate(model, val_loader, criterion, epoch, writer, device='cuda'):
    """验证"""
    model.eval()
    
    total_loss = 0
    loss_components = {
        'dice': 0,
        'iou': 0,
        'bce': 0
    }
    
    print(f'\nRunning validation...')
    print(f'  Validating on {len(val_loader.dataset)} samples...')
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', ncols=100)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            Y_hat, M_coarse = model(images)
            loss, loss_dict = criterion(Y_hat, M_coarse, masks)
            
            total_loss += loss_dict['total']
            loss_components['dice'] += loss_dict['dice_main']
            loss_components['iou'] += loss_dict['iou_main']
            loss_components['bce'] += loss_dict['bce_main']
            
            pbar.set_postfix({
                'Loss': f'{loss_dict["total"]:.4f}',
                'Dice': f'{loss_dict["dice_main"]:.4f}'
            })
    
    # 计算平均
    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches
    
    # TensorBoard
    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Dice', loss_components['dice'], epoch)
        writer.add_scalar('Val/IoU', loss_components['iou'], epoch)
        writer.add_scalar('Val/BCE', loss_components['bce'], epoch)
    
    print(f'  ✓ Validation - Loss: {avg_loss:.4f}, '
          f'Dice: {loss_components["dice"]:.4f}, '
          f'IoU: {loss_components["iou"]:.4f}')
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='E2Net Training with Improved Loss')
    
    # 数据集参数
    parser.add_argument('--train_dataset', type=str, default='../dataset/TrainDataset')
    parser.add_argument('--val_dataset', type=str, default='../dataset/TestDataset/CAMO')
    parser.add_argument('--val_interval', type=int, default=5)
    
    # 模型参数
    parser.add_argument('--encoder_name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--encoder_pretrained', type=str, default='checkpoint/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--freeze_encoder', action='store_true', default=True)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--use_simple_encoder', action='store_true', default=True)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    # 损失函数参数（改进的权重）
    parser.add_argument('--lambda1', type=float, default=2.0,
                       help='Dice loss weight (increased from 1.0)')
    parser.add_argument('--lambda2', type=float, default=1.0,
                       help='BCE loss weight')
    parser.add_argument('--lambda3', type=float, default=0.3,
                       help='Auxiliary loss weight (decreased from 0.5)')
    parser.add_argument('--lambda_edge', type=float, default=0.5,
                       help='Edge loss weight (new)')
    parser.add_argument('--lambda_iou', type=float, default=0.5,
                       help='IoU loss weight (new)')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='checkpoint/E2Net_ImprovedLoss_v3')
    parser.add_argument('--log_dir', type=str, default='logs/E2Net_ImprovedLoss_v3')
    parser.add_argument('--save_interval', type=int, default=10)
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    print("="*70)
    print("E2Net Training with Improved Loss Function")
    print("="*70)
    print("\n🔧 Loss Function Configuration:")
    print(f"  Dice weight:  {args.lambda1} (original: 1.0)")
    print(f"  BCE weight:   {args.lambda2}")
    print(f"  IoU weight:   {args.lambda_iou} (new)")
    print(f"  Edge weight:  {args.lambda_edge} (new)")
    print(f"  Aux weight:   {args.lambda3} (original: 0.5)")
    print("="*70)
    
    # 数据集配置
    train_cfg = dataset.Config(
        datapath=args.train_dataset,
        mode='train',
        batch=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )
    
    # 加载训练数据集
    train_data = dataset.Data(train_cfg, 'E2Net')
    train_loader = DataLoader(
        train_data,
        collate_fn=train_data.collate,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'\n📊 Training dataset size: {len(train_data)}')
    
    # 加载验证数据集
    val_loader = None
    if args.val_dataset and args.val_dataset != 'None':
        val_cfg = dataset.Config(
            datapath=args.val_dataset,
            mode='train',
            batch=1
        )
        val_data = dataset.Data(val_cfg, 'E2Net')
        val_loader = DataLoader(
            val_data,
            collate_fn=val_data.collate,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        print(f'📊 Validation dataset size: {len(val_data)}')
    
    # 构建模型
    print(f'\n🏗️  Building model...')
    
    model = build_e2net(
        cfg=None,
        encoder_name=args.encoder_name,
        encoder_pretrained=args.encoder_pretrained,
        freeze_encoder=args.freeze_encoder,
        feature_dim=args.feature_dim,
        use_simple_encoder=args.use_simple_encoder
    )
    model = model.to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total parameters: {total_params/1e6:.2f}M')
    print(f'  Trainable parameters: {trainable_params/1e6:.2f}M')
    
    # 改进的损失函数
    print(f'\n💡 Setting up improved loss function...')
    criterion = ImprovedLoss(
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda_edge=args.lambda_edge,
        lambda_iou=args.lambda_iou
    )
    print(f'  ✓ Using ImprovedLoss with:')
    print(f'    - Dice loss (weight={args.lambda1})')
    print(f'    - BCE loss (weight={args.lambda2})')
    print(f'    - IoU loss (weight={args.lambda_iou}) [NEW]')
    print(f'    - Edge loss (weight={args.lambda_edge}) [NEW]')
    print(f'    - Auxiliary loss (weight={args.lambda3})')
    
    # 优化器（保持原来的Cosine调度器）
    print(f'\n⚙️  Setting up optimizer and scheduler...')
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # # Cosine退火（保持原样）
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=args.epochs,
    #     eta_min=1e-6
    # )
    # 当前使用 CosineAnnealingLR，考虑改为 ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,        # 降低50%
        patience=10,       # 10个epoch无改善则降低
        min_lr=1e-6,
        verbose=True
    )
    
    print(f'  ✓ Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})')
    print(f'  ✓ Scheduler: CosineAnnealingLR (T_max={args.epochs})')
    
    # 训练配置
    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Val interval:   {args.val_interval}")
    print(f"  Save interval:  {args.save_interval}")
    print("="*70)
    
    # 开始训练
    print(f'\n🚀 Starting training...\n')
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f'\n{"="*70}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*70}')
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                epoch, writer, args.device)
        
        # 验证
        if val_loader is not None and epoch % args.val_interval == 0:
            val_loss = validate(model, val_loader, criterion, epoch, writer, args.device)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.save_dir, 'E2Net_best.pth')
                torch.save(model.state_dict(), best_path)
                print(f'\n✓ Saved best model: {best_path} (Val Loss: {val_loss:.4f})')
                
        # 更新学习率
        metric_for_scheduler = val_loss if val_loss is not None else train_loss
        scheduler.step(metric_for_scheduler)
        # scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # 定期保存
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f'E2Net_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
    
    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'E2Net_final.pth')
    torch.save(model.state_dict(), final_path)
    
    print('\n' + "="*70)
    print('✓ Training completed!')
    print("="*70)
    print(f'  Best model: {best_path} (Val Loss: {best_val_loss:.4f})')
    print(f'  Final model: {final_path}')
    print("="*70)
    
    writer.close()


if __name__ == '__main__':
    main()
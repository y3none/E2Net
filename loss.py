#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的损失函数
主要改进：
1. 修正Dice loss计算
2. 添加边缘增强loss
3. 添加IoU loss
4. 动态权重调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImprovedLoss(nn.Module):
    """改进的组合损失函数"""
    
    def __init__(self, lambda1=2.0, lambda2=1.0, lambda3=0.3, lambda_edge=0.5, lambda_iou=0.5):
        super(ImprovedLoss, self).__init__()
        self.lambda1 = lambda1  # Dice loss权重（增加）
        self.lambda2 = lambda2  # BCE loss权重
        self.lambda3 = lambda3  # Auxiliary loss权重（降低）
        self.lambda_edge = lambda_edge  # 边缘loss权重（新增）
        self.lambda_iou = lambda_iou  # IoU loss权重（新增）
        
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target, smooth=1e-5):
        """
        改进的Dice Loss
        使用logits输入，内部进行sigmoid
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def iou_loss(self, pred, target, smooth=1e-5):
        """IoU Loss"""
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return 1 - iou
    
    def edge_loss(self, pred, target):
        """
        边缘增强损失
        使用Sobel算子提取边缘，然后计算边缘区域的BCE
        """
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        # 提取目标边缘
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)
        
        # 归一化并二值化 - 确保在[0,1]范围内
        target_edge_max = target_edge.max()
        if target_edge_max > 1e-6:
            target_edge = target_edge / target_edge_max
        target_edge = (target_edge > 0.1).float()
        target_edge = torch.clamp(target_edge, 0.0, 1.0)  # 严格限制
        
        # 预测边缘
        pred_sigmoid = torch.sigmoid(pred)
        pred_edge_x = F.conv2d(pred_sigmoid, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_sigmoid, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        
        # 归一化到[0,1]
        pred_edge_max = pred_edge.max()
        if pred_edge_max > 1e-6:
            pred_edge = pred_edge / pred_edge_max
        pred_edge = torch.clamp(pred_edge, 0.0, 1.0)  # 严格限制
        
        # 边缘区域的BCE（不使用weight避免问题）
        edge_loss = F.binary_cross_entropy(pred_edge, target_edge, reduction='mean')
        
        return edge_loss
    
    def forward(self, pred_main, pred_coarse, target):
        """
        计算总损失
        
        Args:
            pred_main: 主预测 (B, 1, H, W) - logits
            pred_coarse: 粗预测 (B, 1, H, W) - logits
            target: 目标 (B, 1, H, W) - [0, 1]
        
        Returns:
            total_loss, loss_dict
        """
        # 主预测的损失
        dice_main = self.dice_loss(pred_main, target)
        bce_main = self.bce(pred_main, target)
        iou_main = self.iou_loss(pred_main, target)
        edge_main = self.edge_loss(pred_main, target)
        
        # 辅助预测的损失
        dice_aux = self.dice_loss(pred_coarse, target)
        bce_aux = self.bce(pred_coarse, target)
        
        # 组合损失
        main_loss = (self.lambda1 * dice_main + 
                    self.lambda2 * bce_main + 
                    self.lambda_iou * iou_main +
                    self.lambda_edge * edge_main)
        
        aux_loss = self.lambda3 * (dice_aux + bce_aux)
        
        total_loss = main_loss + aux_loss
        
        # 返回详细信息（用于监控）
        loss_dict = {
            'total': total_loss.item(),
            'dice_main': dice_main.item(),
            'bce_main': bce_main.item(),
            'iou_main': iou_main.item(),
            'edge_main': edge_main.item(),
            'aux': aux_loss.item()
        }
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """Focal Loss - 用于处理难例样本"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - logits
            target: (B, 1, H, W) - [0, 1]
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Focal weight
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        
        # BCE with focal weight
        bce = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
        focal_loss = self.alpha * focal_weight * bce
        
        return focal_loss.mean()


class DynamicWeightedLoss(nn.Module):
    """动态调整损失权重"""
    
    def __init__(self, base_loss):
        super(DynamicWeightedLoss, self).__init__()
        self.base_loss = base_loss
        self.epoch = 0
    
    def update_epoch(self, epoch):
        """更新epoch，动态调整权重"""
        self.epoch = epoch
        
        # 前期注重Dice和IoU，后期注重边缘
        if epoch < 30:
            self.base_loss.lambda1 = 2.0  # Dice
            self.base_loss.lambda_edge = 0.3  # Edge
        elif epoch < 60:
            self.base_loss.lambda1 = 1.5
            self.base_loss.lambda_edge = 0.5
        else:
            self.base_loss.lambda1 = 1.0
            self.base_loss.lambda_edge = 0.8  # 后期强调边缘
    
    def forward(self, pred_main, pred_coarse, target):
        return self.base_loss(pred_main, pred_coarse, target)


if __name__ == '__main__':
    # 测试损失函数
    print("Testing Improved Loss Function...")
    
    # 创建测试数据
    B, H, W = 2, 384, 384
    pred_main = torch.randn(B, 1, H, W)  # logits
    pred_coarse = torch.randn(B, 1, H, W)  # logits
    target = torch.rand(B, 1, H, W)  # [0, 1]
    
    # 测试改进的损失
    criterion = ImprovedLoss()
    loss, loss_dict = criterion(pred_main, pred_coarse, target)
    
    print("\nLoss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTotal Loss: {loss.item():.4f}")
    print("✓ Loss function test passed!")
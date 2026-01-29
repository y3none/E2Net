import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    通道注意力模块 (SENet/ECANet风格)
    用于增强重要特征通道的响应
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CAEM(nn.Module):
    """
    Cross-Scale Feature Alignment and Enhancement Module
    跨尺度特征对齐与增强模块
    
    功能：
    1. 将不同尺度的DINOv3特征对齐到统一分辨率
    2. 通过渐进式上采样和融合，将深层语义注入浅层细节
    3. 使用通道注意力增强关键特征响应
    """
    def __init__(self, in_channels=768, out_channels=128):
        super(CAEM, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1x1卷积用于通道降维，将DINOv3的特征维度(768)降到目标维度(128)
        self.conv_F1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_F2 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_F3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_F4 = nn.Conv2d(in_channels, out_channels, 1)
        
        # 融合模块：用于特征融合后的处理
        self.fuse_conv_4to3 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fuse_conv_3to2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fuse_conv_2to1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力模块，用于增强融合后的特征
        self.ca_G1 = ChannelAttention(out_channels)
        self.ca_G2 = ChannelAttention(out_channels)
        self.ca_G3 = ChannelAttention(out_channels)
        self.ca_G4 = ChannelAttention(out_channels)
        
    def forward(self, features):
        """
        Args:
            features: [F1, F2, F3, F4] from DINOv3
                F1: [B, C, H/4, W/4]   - 浅层高分辨率
                F2: [B, C, H/8, W/8]
                F3: [B, C, H/16, W/16]
                F4: [B, C, H/32, W/32] - 深层低分辨率
        Returns:
            aligned_features: [G1, G2, G3, G4]
                统一到最高分辨率，每层都融合了不同深度的语义
        """
        F1, F2, F3, F4 = features
        
        # Step 1: 通道降维
        F1 = self.conv_F1(F1)  # [B, 128, H/4, W/4]
        F2 = self.conv_F2(F2)  # [B, 128, H/8, W/8]
        F3 = self.conv_F3(F3)  # [B, 128, H/16, W/16]
        F4 = self.conv_F4(F4)  # [B, 128, H/32, W/32]
        
        # Step 2: 渐进式上采样与融合
        # 从最深层F4开始，逐步向浅层融合
        
        # F4 -> F3: 上采样F4并与F3融合
        F4_up = F.interpolate(F4, size=F3.size()[2:], mode='bilinear', align_corners=False)
        G3_fused = torch.cat([F4_up, F3], dim=1)  # [B, 256, H/16, W/16]
        G3 = self.fuse_conv_4to3(G3_fused)         # [B, 128, H/16, W/16]
        
        # G3 -> F2: 上采样融合后的G3并与F2融合
        G3_up = F.interpolate(G3, size=F2.size()[2:], mode='bilinear', align_corners=False)
        G2_fused = torch.cat([G3_up, F2], dim=1)   # [B, 256, H/8, W/8]
        G2 = self.fuse_conv_3to2(G2_fused)         # [B, 128, H/8, W/8]
        
        # G2 -> F1: 上采样融合后的G2并与F1融合
        G2_up = F.interpolate(G2, size=F1.size()[2:], mode='bilinear', align_corners=False)
        G1_fused = torch.cat([G2_up, F1], dim=1)   # [B, 256, H/4, W/4]
        G1 = self.fuse_conv_2to1(G1_fused)         # [B, 128, H/4, W/4]
        
        # 保持G4为降维后的F4
        G4 = F4
        
        # Step 3: 统一分辨率到最高分辨率 (H/4, W/4)
        target_size = G1.size()[2:]
        G2 = F.interpolate(G2, size=target_size, mode='bilinear', align_corners=False)
        G3 = F.interpolate(G3, size=target_size, mode='bilinear', align_corners=False)
        G4 = F.interpolate(G4, size=target_size, mode='bilinear', align_corners=False)
        
        # Step 4: 通道注意力增强
        G1 = self.ca_G1(G1)
        G2 = self.ca_G2(G2)
        G3 = self.ca_G3(G3)
        G4 = self.ca_G4(G4)
        
        return [G1, G2, G3, G4]
    
    def initialize(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

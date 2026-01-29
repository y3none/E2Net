import torch
import torch.nn as nn
import torch.nn.functional as F


class LFSM(nn.Module):
    """
    Lateral Fovea Search Module (侧视野搜索模块)
    Stage 1: 模拟老鹰的广域搜索，生成粗略的显著图
    
    功能：
    - 在最具全局语义的特征G4上应用自注意力
    - 捕获全局上下文关系
    - 生成初始粗糙显著图 M_coarse
    """
    def __init__(self, dim=128, num_heads=8):
        super(LFSM, self).__init__()
        
        # 全局上下文分析器 (简化的ViT块)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP for feature enhancement
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # 粗糙显著图生成头
        self.coarse_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, G4):
        """
        Args:
            G4: 最深层融合特征 [B, C, H, W]
        Returns:
            M_coarse: 粗糙显著图 [B, 1, H, W]
        """
        B, C, H, W = G4.shape
        
        # 将空间维度展平，准备进行自注意力计算
        x = G4.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 自注意力模块 (捕获全局关系)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # 恢复空间维度
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # 生成粗糙显著图
        M_coarse = self.coarse_head(x)
        
        return M_coarse
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class CFZM(nn.Module):
    """
    Central Fovea Zooming Module (中央视野聚焦模块)
    Stage 2: 模拟老鹰的动态聚焦，增强关键区域
    
    功能：
    - 使用M_coarse生成多尺度注意力权重
    - 对所有层级特征进行动态聚焦增强
    - 输出增强后的多尺度特征 {G'1, G'2, G'3, G'4}
    """
    def __init__(self, dim=128):
        super(CFZM, self).__init__()
        
        # 特征调制网络（针对每个层级）
        self.modulate_G1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        self.modulate_G2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        self.modulate_G3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        self.modulate_G4 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, aligned_features, M_coarse):
        """
        Args:
            aligned_features: [G1, G2, G3, G4] from CAEM
            M_coarse: 粗糙显著图 [B, 1, H, W] from LFSM
        Returns:
            enhanced_features: [G'1, G'2, G'3, G'4]
        """
        G1, G2, G3, G4 = aligned_features
        
        # 将M_coarse上采样到与每个Gi相同的尺寸，得到注意力权重Ai
        A1 = F.interpolate(M_coarse, size=G1.size()[2:], mode='bilinear', align_corners=False)
        A2 = F.interpolate(M_coarse, size=G2.size()[2:], mode='bilinear', align_corners=False)
        A3 = F.interpolate(M_coarse, size=G3.size()[2:], mode='bilinear', align_corners=False)
        A4 = F.interpolate(M_coarse, size=G4.size()[2:], mode='bilinear', align_corners=False)
        
        # 动态聚焦：G'i = Gi ⊗ σ(Ai) + Gi (残差连接)
        # ⊗ 表示逐元素相乘
        G1_prime = self.modulate_G1(G1 * A1 + G1)
        G2_prime = self.modulate_G2(G2 * A2 + G2)
        G3_prime = self.modulate_G3(G3 * A3 + G3)
        G4_prime = self.modulate_G4(G4 * A4 + G4)
        
        return [G1_prime, G2_prime, G3_prime, G4_prime]
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CCM(nn.Module):
    """
    Cognitive Confirmation Module (认知确认模块)
    Stage 3: 模拟老鹰的精细判断，通过交叉注意力整合信息
    
    功能：
    - Query来自细节特征G'1，Key/Value来自语义特征G'4
    - 门控交叉注意力机制，防止噪声干扰
    - 渐进式解码器细化分割结果
    """
    def __init__(self, dim=128, num_heads=8):
        super(CCM, self).__init__()
        
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        # Query, Key, Value 投影
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)  # 修复：添加kernel_size=1
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )
        
        # 渐进式解码器
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(3)  # 3次上采样
        ])
        
        # 特征融合
        self.fuse_G2 = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.fuse_G3 = nn.Conv2d(dim * 2, dim, 3, padding=1)
        
        # 最终预测头
        self.final_pred = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1)
        )
        
    def forward(self, enhanced_features):
        """
        Args:
            enhanced_features: [G'1, G'2, G'3, G'4] from CFZM
        Returns:
            Y_hat: 最终高分辨率分割图 [B, 1, H, W]
        """
        G1_prime, G2_prime, G3_prime, G4_prime = enhanced_features
        
        B, C, H, W = G1_prime.shape
        
        # 门控交叉注意力
        # Query: 细节特征 G'1
        # Key, Value: 语义特征 G'4
        Q = self.q_proj(G1_prime)  # [B, C, H, W]
        K = self.k_proj(G4_prime)  # [B, C, H', W']
        V = self.v_proj(G4_prime)  # [B, C, H', W']
        
        # 上采样K和V到与Q相同的尺寸
        K = F.interpolate(K, size=(H, W), mode='bilinear', align_corners=False)
        V = F.interpolate(V, size=(H, W), mode='bilinear', align_corners=False)
        
        # 计算注意力
        # Reshape for multi-head attention
        Q = Q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        K = K.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        V = V.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # 门控机制
        gate_input = torch.cat([G1_prime, out], dim=1)
        gate_weight = self.gate(gate_input)
        fused = G1_prime + gate_weight * out
        
        # 渐进式解码
        # 融合G'3
        x = self.decoder[0](fused)  # 上采样
        G3_up = F.interpolate(G3_prime, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = self.fuse_G3(torch.cat([x, G3_up], dim=1))
        
        # 融合G'2
        x = self.decoder[1](x)  # 上采样
        G2_up = F.interpolate(G2_prime, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = self.fuse_G2(torch.cat([x, G2_up], dim=1))
        
        # 最后一次上采样
        x = self.decoder[2](x)
        
        # 最终预测
        Y_hat = self.final_pred(x)
        
        return Y_hat
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
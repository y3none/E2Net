import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3_encoder import build_dinov3_encoder
from caem import CAEM
from dual_fovea_attention import LFSM, CFZM, CCM


class E2Net(nn.Module):
    """
    EagleEyeNet (E2Net): A Bio-Inspired Decision Cortex for Camouflaged Object Detection
    
    使用HuggingFace的DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m) 作为视觉主干
    
    架构组成：
    1. DINOv3 Encoder: 预训练的视觉主干网络（冻结）
       - Model: facebook/dinov3-vitb16-pretrain-lvd1689m
       - Architecture: ViT-B/16
       - Parameters: 86M
       - Patch size: 16
       - Embed dim: 768
       - Register tokens: 4
       - Position encoding: RoPE
    
    2. CAEM: 跨尺度特征对齐与增强模块
       - 将768维特征降到128维
       - 渐进式融合多尺度特征
    
    3. Dual Fovea Attention Pipeline:
       - Stage 1 (LFSM): 广域搜索，生成粗略显著图
       - Stage 2 (CFZM): 动态聚焦，增强关键区域
       - Stage 3 (CCM): 认知确认，精细分割
    """
    def __init__(self, cfg=None, 
                 encoder_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
                 encoder_pretrained=None,
                 freeze_encoder=True,
                 feature_dim=256,
                 use_simple_encoder=False):
        super(E2Net, self).__init__()
        
        self.cfg = cfg
        self.feature_dim = feature_dim
        
        print("="*60)
        print("Building E2Net with DINOv3...")
        print("="*60)
        
        # 1. DINOv3 Encoder (预训练视觉皮层)
        print("\n[1/4] Building DINOv3 Encoder...")
        self.encoder = build_dinov3_encoder(
            model_name=encoder_name,
            pretrained_path=encoder_pretrained,
            freeze=freeze_encoder,
            simple=use_simple_encoder
        )
        
        # 2. CAEM (跨尺度对齐与增强)
        # DINOv3输出768维，降到feature_dim维
        print(f"\n[2/4] Building CAEM (768 → {feature_dim})...")
        self.caem = CAEM(in_channels=768, out_channels=feature_dim)
        
        # 3. Dual Fovea Attention Pipeline
        print("\n[3/4] Building Dual Fovea Attention Pipeline...")
        
        # Stage 1: Lateral Fovea Search Module
        self.lfsm = LFSM(dim=feature_dim, num_heads=8)
        print("  ✓ LFSM (Stage 1: Lateral Fovea Search)")
        
        # Stage 2: Central Fovea Zooming Module
        self.cfzm = CFZM(dim=feature_dim)
        print("  ✓ CFZM (Stage 2: Central Fovea Zooming)")
        
        # Stage 3: Cognitive Confirmation Module
        self.ccm = CCM(dim=feature_dim, num_heads=8)
        print("  ✓ CCM (Stage 3: Cognitive Confirmation)")
        
        # 4. 初始化除encoder外的所有模块
        print("\n[4/4] Initializing decision modules...")
        self.initialize()
        
        print("\n" + "="*60)
        print("E2Net model built successfully!")
        print("="*60)
        
    def forward(self, x, shape=None):
        """
        Args:
            x: Input image [B, 3, H, W]
            shape: Target output shape (H, W)
        Returns:
            Y_hat: 最终分割预测 [B, 1, H, W]
            M_coarse: 粗略显著图 (用于辅助损失)
        """
        if shape is None:
            shape = x.size()[2:]
        
        # Step 1: DINOv3特征提取
        # 获取4个不同层级的特征 [F1, F2, F3, F4]
        features = self.encoder(x)
        
        # Step 2: CAEM - 特征对齐与增强
        # 输出统一分辨率的多层级特征 [G1, G2, G3, G4]
        aligned_features = self.caem(features)
        
        # Step 3: Stage 1 - LFSM (Lateral Fovea Search)
        # 在最深层特征G4上进行全局搜索
        # 输出粗略显著图 M_coarse
        M_coarse = self.lfsm(aligned_features[3])  # G4
        
        # Step 4: Stage 2 - CFZM (Central Fovea Zooming)
        # 使用M_coarse动态聚焦所有层级特征
        # 输出增强特征 [G'1, G'2, G'3, G'4]
        enhanced_features = self.cfzm(aligned_features, M_coarse)
        
        # Step 5: Stage 3 - CCM (Cognitive Confirmation)
        # 通过门控交叉注意力和渐进解码生成最终分割
        Y_hat = self.ccm(enhanced_features)
        
        # 上采样到目标尺寸
        Y_hat = F.interpolate(Y_hat, size=shape, mode='bilinear', align_corners=False)
        M_coarse = F.interpolate(M_coarse, size=shape, mode='bilinear', align_corners=False)
        
        return Y_hat, M_coarse
    
    def initialize(self):
        """初始化模型权重（除encoder外）"""
        # 初始化CAEM
        self.caem.initialize()
        
        # 初始化Dual Fovea Attention Pipeline
        self.lfsm.initialize()
        self.cfzm.initialize()
        self.ccm.initialize()
        
        # 如果有预训练权重，加载
        if self.cfg is not None and hasattr(self.cfg, 'snapshot') and self.cfg.snapshot:
            try:
                state_dict = torch.load(self.cfg.snapshot, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                print(f'  ✓ Loaded E2Net checkpoint from {self.cfg.snapshot}')
            except Exception as e:
                print(f'  ✗ Failed to load checkpoint: {e}')
                print('  → Initializing from scratch...')


def build_e2net(cfg=None, **kwargs):
    """
    构建E2Net模型
    
    Args:
        cfg: 配置对象
        **kwargs: 其他参数
            - encoder_name: DINOv3模型名称（默认: 'facebook/dinov3-vitb16-pretrain-lvd1689m'）
            - encoder_pretrained: DINOv3预训练权重路径
            - freeze_encoder: 是否冻结encoder
            - feature_dim: 特征维度
            - use_simple_encoder: 是否使用简化版编码器
    """
    model = E2Net(cfg=cfg, **kwargs)
    return model


if __name__ == '__main__':
    # 测试模型
    print("\n" + "="*60)
    print("Testing E2Net with DINOv3...")
    print("="*60 + "\n")
    
    # 创建模型
    model = build_e2net(
        encoder_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        encoder_pretrained=None,  # 设置为本地路径或None
        freeze_encoder=True,
        feature_dim=256,
        use_simple_encoder=False
    )
    
    # 测试前向传播
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 384, 384)
    print(f"Input shape: {x.shape}")
    
    Y_hat, M_coarse = model(x)
    
    print(f"Output Y_hat shape: {Y_hat.shape}")
    print(f"Output M_coarse shape: {M_coarse.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total_params / 1e6:>8.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:>8.2f}M")
    print(f"  Frozen parameters:    {frozen_params / 1e6:>8.2f}M")
    
    print("\n" + "="*60)
    print("E2Net test passed! ✓")
    print("="*60)

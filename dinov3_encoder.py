import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor


class DINOv3Encoder(nn.Module):
    """
    DINOv3 Encoder using HuggingFace transformers
    
    Model: facebook/dinov3-vitb16-pretrain-lvd1689m
    - Architecture: ViT-B/16
    - Parameters: 86M
    - Patch size: 16
    - Embedding dimension: 768
    - Register tokens: 4
    - Position encoding: RoPE (Rotary Position Embedding)
    - Heads: 12
    - FFN: MLP
    
    For a 224x224 image:
    - 1 class token + 4 register tokens + 196 patch tokens = 201 tokens
    
    Key difference from DINOv2:
    - Uses RoPE instead of absolute position embeddings
    - Has 4 register tokens for better dense prediction tasks
    """
    def __init__(self, model_name='facebook/dinov3-vitb16-pretrain-lvd1689m', 
                 freeze=True, 
                 pretrained_path=None):
        super(DINOv3Encoder, self).__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Load DINOv3 model from HuggingFace
        if pretrained_path:
            # 从本地路径加载
            print(f"Loading DINOv3 from local path: {pretrained_path}")
            self.backbone = AutoModel.from_pretrained(pretrained_path)
            self.processor = AutoImageProcessor.from_pretrained(pretrained_path)
        else:
            # 从HuggingFace Hub加载
            print(f"Loading DINOv3 from HuggingFace: {model_name}")
            self.backbone = AutoModel.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # DINOv3-Base配置
        self.patch_size = 16  # ViT-B/16
        self.embed_dim = 768
        self.num_layers = 12
        self.num_register_tokens = 4  # DINOv3特有
        
        # 我们提取4个中间层的特征
        # 层索引: 3, 6, 9, 12
        self.extract_layers = [2, 5, 8, 11]  # Python 0-based indexing
        
        if freeze:
            self._freeze_backbone()
        
        print(f"DINOv3 Encoder initialized!")
        print(f"  Model: {model_name}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Register tokens: {self.num_register_tokens}")
        print(f"  Freeze: {freeze}")
        
    def _freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  DINOv3 encoder frozen ✓")
    
    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W]
        Returns:
            features: List of 4 feature maps [F1, F2, F3, F4]
                F1: [B, 768, H/4,  W/4]   - 早期特征（层3）
                F2: [B, 768, H/8,  W/8]   - 中层特征（层6）
                F3: [B, 768, H/16, W/16]  - 中层特征（层9）
                F4: [B, 768, H/32, W/32]  - 深层特征（层12）
        """
        B, C, H, W = x.shape
        
        # DINOv3前向传播，获取所有hidden states
        outputs = self.backbone(
            x,
            output_hidden_states=True,
            return_dict=True
        )
        
        # hidden_states: (num_layers+1) x [B, num_tokens, embed_dim]
        # num_tokens = 1(cls) + 4(register) + num_patches
        hidden_states = outputs.hidden_states
        
        # 提取我们需要的4个层的特征
        features = []
        
        for i, layer_idx in enumerate(self.extract_layers):
            # 获取该层的hidden state: [B, num_tokens, 768]
            feat = hidden_states[layer_idx + 1]  # +1因为有initial embedding
            
            # DINOv3: 移除CLS token (第1个) 和 register tokens (第2-5个)
            # 只保留patch tokens
            feat = feat[:, 1 + self.num_register_tokens:, :]  # [B, num_patches, 768]
            
            # 计算patch网格尺寸
            num_patches = feat.shape[1]
            patch_h = patch_w = int(num_patches ** 0.5)
            
            # Reshape到2D特征图
            feat = feat.permute(0, 2, 1).reshape(B, self.embed_dim, patch_h, patch_w)
            # feat: [B, 768, h, w], where h = w = H/patch_size
            
            # 插值到期望的分辨率
            # 期望的输出分辨率：H/4, H/8, H/16, H/32
            target_sizes = [H//4, H//8, H//16, H//32]
            target_size = target_sizes[i]
            
            feat = F.interpolate(
                feat, 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            features.append(feat)
        
        return features


class DINOv3EncoderSimple(nn.Module):
    """
    简化版DINOv3编码器
    直接使用不同层的特征，不进行额外的分辨率调整
    更快，但可能精度略低
    """
    def __init__(self, model_name='facebook/dinov3-vitb16-pretrain-lvd1689m', 
                 freeze=True, 
                 pretrained_path=None):
        super(DINOv3EncoderSimple, self).__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Load DINOv3 model
        if pretrained_path:
            print(f"Loading DINOv3 from: {pretrained_path}")
            self.backbone = AutoModel.from_pretrained(pretrained_path)
            self.processor = AutoImageProcessor.from_pretrained(pretrained_path)
        else:
            print(f"Loading DINOv3 from HuggingFace: {model_name}")
            self.backbone = AutoModel.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        self.patch_size = 16
        self.embed_dim = 768
        self.num_register_tokens = 4
        
        # 提取层索引
        self.extract_layers = [2, 5, 8, 11]
        
        if freeze:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  DINOv3 encoder frozen ✓")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [F1, F2, F3, F4], 每个都是 [B, 768, h, w]
                其中 h = w = H / patch_size
        """
        B, C, H, W = x.shape
        
        # 获取所有hidden states
        outputs = self.backbone(
            x,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states
        features = []
        
        # patch网格尺寸
        patch_h = patch_w = H // self.patch_size
        
        for layer_idx in self.extract_layers:
            # 获取hidden state并移除CLS + register tokens
            feat = hidden_states[layer_idx + 1][:, 1 + self.num_register_tokens:, :]  
            # [B, num_patches, 768]
            
            # Reshape到2D
            feat = feat.permute(0, 2, 1).reshape(B, self.embed_dim, patch_h, patch_w)
            
            features.append(feat)
        
        return features


def build_dinov3_encoder(model_name='facebook/dinov3-vitb16-pretrain-lvd1689m', 
                         pretrained_path=None, 
                         freeze=True,
                         simple=False):
    """
    构建DINOv3编码器
    
    Args:
        model_name: HuggingFace模型名称
            - facebook/dinov3-vits16-pretrain-lvd1689m (Small, 21M)
            - facebook/dinov3-vitb16-pretrain-lvd1689m (Base, 86M) ← 推荐
            - facebook/dinov3-vitl16-pretrain-lvd1689m (Large, 300M)
            - facebook/dinov3-vith16plus-pretrain-lvd1689m (Huge+, 840M)
        pretrained_path: 本地预训练权重路径
        freeze: 是否冻结编码器
        simple: 是否使用简化版（不调整分辨率）
    """
    if simple:
        encoder = DINOv3EncoderSimple(
            model_name=model_name,
            freeze=freeze,
            pretrained_path=pretrained_path
        )
    else:
        encoder = DINOv3Encoder(
            model_name=model_name,
            freeze=freeze,
            pretrained_path=pretrained_path
        )
    
    return encoder


if __name__ == '__main__':
    # 测试编码器
    print("="*60)
    print("Testing DINOv3 Encoder...")
    print("="*60)
    
    # 创建编码器
    encoder = build_dinov3_encoder(
        model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        pretrained_path=None,  # 设置为本地路径或None
        freeze=True,
        simple=False
    )
    
    # 测试前向传播
    x = torch.randn(2, 3, 384, 384)
    print(f"\nInput shape: {x.shape}")
    
    features = encoder(x)
    
    print(f"\nOutput features:")
    for i, feat in enumerate(features):
        print(f"  F{i+1}: {feat.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("DINOv3 Encoder test passed!")
    print("="*60)

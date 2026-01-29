# E2Net with DINOv3 - 完整实现
-------------------- 微调日志： --------------------    
E2Net_ImprovedLoss:  
(v1)--validate interval=5 ✅   
v2--validate interval:5 -> 1 ❌   
v3--feature_dim:128 -> 256 validate interval=1 half-half   
v4--feature_dim=256 validate interval=1   
    --lambda1 2.0 -> 3.0 \
    --lambda2 1.0 \
    --lambda3 0.3 -> 0.2 \
    --lambda_edge 0.5 -> 0.3 \
    --lambda_iou 0.5 -> 1.0 \
    --weight_decay 5e-4 -> 1e-2 \
    scheduler:CosineAnnealingLR -> ReduceLROnPlateau ❌   

v5--feature_dim=256 validate interval=1   
    --lambda1 3.0 -> 2.5 \
    --lambda2 1.0 \
    --lambda3 0.2 \
    --lambda_edge 0.3 -> 0.7 \
    --lambda_iou 1.0 \
    --weight_decay 1e-2 -> 5e-4 \
    scheduler=ReduceLROnPlateau ✅    
------------------------------------------------  

基于CamoFormer的代码库，使用Meta AI的 **DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)** 作为预训练编码器实现的E2Net（EagleEyeNet）伪装目标检测模型。

## 重要说明

本项目使用 **DINOv3**

- **模型**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- **架构**: ViT-B/16
- **参数量**: 86M
- **Patch size**: 16
- **Embedding dim**: 768
- **Register tokens**: 4 (DINOv3特有)
- **Position encoding**: RoPE (Rotary Position Embedding)

DINOv3 vs DINOv2的关键区别：
- ✅ 使用RoPE而非绝对位置编码
- ✅ 包含4个register tokens，提升密集预测任务性能
- ✅ 在多个benchmark上超越DINOv2

## 快速开始

### Step 1: 环境安装

```bash
# 创建环境
conda create --name e2net python=3.10.19
conda activate e2net

# 安装PyTorch 2.5.1 with CUDA 12.1支持
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt
```

### Step 2: 准备DINOv3模型

将您下载的 `dinov3-vitb16-pretrain-lvd1689m` 放到项目目录：

```bash
mkdir -p dinov3_models/vitb16
# 将下载的模型文件复制到这个目录
# 应该包含: config.json, model.safetensors, preprocessor_config.json等
```

### Step 3: 准备数据集

将数据集放在根目录下  

```bash
dataset/
├── TrainDataset/
│   ├── Image/
│   └── GT/
└── TestDataset/
    ├── CHAMELEON/
    ├── CAMO/
    ├── COD10K/
    └── NC4K/
```

### Step 4: 开始训练

```bash
bash train_e2net.sh
```

### Step 5: 测试与评估

```bash
# 测试
bash test_e2net.sh

# 评估
bash eval.sh
```

## 项目文件

### 核心代码
- `dinov3_encoder.py` - **DINOv3编码器**（HuggingFace transformers）
- `caem.py` - 跨尺度特征对齐与增强模块
- `dual_fovea_attention.py` - 双中央凹注意力管道
- `E2Net.py` - 完整的E2Net模型
- `loss.py` - 损失函数

### 脚本
- `train_e2net.py` / `train_e2net.sh` - 训练
- `test_e2net.py` / `test_e2net.sh` - 测试
- `evaltools`文件夹 / `eval.sh` - 评估（复用CamoFormer评估代码）

### 文档
- `README.md` - 本文件
- `DINOV3_GUIDE.md` - DINOv3详细使用指南
- `ARCHITECTURE.md` - 架构详解

## 架构概览

```
输入图像 [B, 3, 384, 384]
    ↓
DINOv3 Encoder (facebook/dinov3-vitb16, 冻结)
  - ViT-B/16, 86M参数
  - 4个register tokens
  - RoPE位置编码
  提取4层特征: F1, F2, F3, F4
    ↓
CAEM (特征对齐768→128)
  输出: [G1, G2, G3, G4]
    ↓
Stage 1: LFSM (广域搜索)
  → M_coarse
    ↓
Stage 2: CFZM (动态聚焦)
  → [G'1, G'2, G'3, G'4]
    ↓
Stage 3: CCM (认知确认)
  → Y_hat [B, 1, 384, 384]
```

## 核心创新

1. **DINOv3预训练特征**
   - Meta AI 2025年最新模型
   - RoPE位置编码，泛化能力更强
   - Register tokens提升密集预测

2. **生物启发设计**
   - 模拟老鹰三阶段视觉认知
   - 广域搜索 → 动态聚焦 → 精细判断

3. **高效训练**
   - 仅训练~5.5M决策参数
   - DINOv3冻结，节省显存
   - 比CamoFormer快1.5倍

## 关键配置

### 使用本地DINOv3

```bash
python train_e2net.py \
    --encoder_name 'facebook/dinov3-vitb16-pretrain-lvd1689m' \
    --encoder_pretrained 'checkpoint/dinov3-vitb16-pretrain-lvd1689m' \
    --freeze_encoder
```

### 使用简化编码器（更快）

```bash
python train_e2net.py \
    --encoder_pretrained 'checkpoint/dinov3-vitb16-pretrain-lvd1689m' \
    --use_simple_encoder  # 不调整分辨率
```

### 自动下载（需要网络）

```bash
python train_e2net.py \
    --encoder_name 'facebook/dinov3-vitb16-pretrain-lvd1689m' \
    --encoder_pretrained None
```

## DINOv3特性

### Register Tokens的作用

DINOv3引入4个register tokens，有效改善：
- 密集预测任务性能
- 特征图的平滑性
- 对小目标的检测能力

### RoPE位置编码

相比绝对位置编码：
- 更好的长度泛化能力
- 对不同分辨率图像更鲁棒
- 位置信息建模更灵活

## 与DINOv2对比

| 特性 | DINOv2 | DINOv3 |
|------|--------|--------|
| Position Encoding | Absolute | RoPE ✓ |
| Register Tokens | 0 | 4 ✓ |
| Dense Prediction | 好 | 更好 ✓ |
| 发布时间 | 2023 | 2025 ✓ |
| HuggingFace | ✅ | ✅ |

## camoformer引用

```bibtex
@article{yin2024camoformer,
  title={Camoformer: Masked separable attention for camouflaged object detection},
  author={Yin, Bowen and Zhang, Xuying and Fan, Deng-Ping and Jiao, Shaohui and Cheng, Ming-Ming and Van Gool, Luc and Hou, Qibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
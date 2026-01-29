import os
import sys
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
import argparse

# 导入数据集
import dataset

# 导入E2Net
from E2Net import build_e2net


def test(args):
    """测试E2Net模型"""
    
    # 测试数据集列表
    test_datasets = ['CHAMELEON', 'COD10K', 'NC4K', 'CAMO']
    
    for dataset_name in test_datasets:
        print(f'\n{"="*60}')
        print(f'Testing on {dataset_name}...')
        print(f'{"="*60}')
        
        dataset_path = os.path.join(args.test_dataset, dataset_name)
        
        # 配置
        cfg = dataset.Config(
            datapath=dataset_path,
            snapshot=args.checkpoint,
            mode='test'
        )
        
        # 加载数据
        test_data = dataset.Data(cfg, 'E2Net')
        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        # 构建模型
        model = build_e2net(
            cfg=None,
            encoder_name=args.encoder_name,
            encoder_pretrained=args.encoder_pretrained,
            freeze_encoder=True,
            feature_dim=args.feature_dim,
            use_simple_encoder=args.use_simple_encoder
        )
        
        # 加载权重
        if args.checkpoint:
            print(f'Loading checkpoint: {args.checkpoint}')
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            print('Warning: No checkpoint provided, using randomly initialized model')
        
        # 设置为评估模式
        model.eval()
        
        # 使用CPU或CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f'Using device: {device}')
        print(f'Number of test images: {len(test_data)}')
        
        # 预测输出目录
        output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试
        with torch.no_grad():
            for idx, (image, shape, name) in enumerate(test_loader):
                image = image.to(device)
                H, W = shape
                
                # 前向传播
                Y_hat, M_coarse = model(image, shape=(H.item(), W.item()))
                
                # 获取预测结果
                pred = torch.sigmoid(Y_hat[0, 0]).cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                
                # 保存预测结果
                save_path = os.path.join(output_dir, name[0].replace('.jpg', '.png'))
                cv2.imwrite(save_path, pred)
                
                if (idx + 1) % 50 == 0:
                    print(f'Processed {idx + 1}/{len(test_data)} images')
        
        print(f'Predictions saved to: {output_dir}')
    
    print(f'\n{"="*60}')
    print('Testing completed!')
    print(f'{"="*60}')


def main():
    parser = argparse.ArgumentParser(description='Test E2Net on camouflaged object detection datasets')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, default='checkpoint/E2Net/E2Net_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--encoder_name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m',
                        help='DINOv3 encoder variant (HuggingFace model name)')
    parser.add_argument('--encoder_pretrained', type=str, default=None,
                        help='Path to DINOv3 pretrained weights (local directory)')
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='Feature dimension')
    parser.add_argument('--use_simple_encoder', action='store_true', default=False,
                        help='Use simplified encoder')
    
    # 数据集参数
    parser.add_argument('--test_dataset', type=str, default='../dataset/TestDataset',
                        help='Path to test dataset root')
    parser.add_argument('--output_dir', type=str, default='output/Prediction/E2Net-test',
                        help='Directory to save predictions')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行测试
    test(args)


if __name__ == '__main__':
    main()

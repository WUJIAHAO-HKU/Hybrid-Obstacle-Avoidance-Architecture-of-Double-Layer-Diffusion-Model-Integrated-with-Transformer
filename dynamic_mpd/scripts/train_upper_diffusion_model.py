"""
上层扩散模型训练脚本 - 用于正式仿真

训练Transformer障碍物轨迹预测模型并保存checkpoint
供 rosorin_lidar_navigation.py 等仿真脚本加载使用

Author: Dynamic MPD Project
Date: 2026-01-30
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

from src.transformer_obstacle_diffusion import (
    TransformerObstacleDiffusionWrapper,
    TransformerDiffusionConfig
)
from src.complex_obstacle_data import generate_complex_training_data


def train_and_save_model(args):
    """训练并保存模型"""
    
    print("=" * 70)
    print("  Upper Diffusion Model Training")
    print("  Transformer-based Obstacle Trajectory Prediction")
    print("=" * 70)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\n  Device: {device}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # 创建模型配置
    config = TransformerDiffusionConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        diffusion_steps=args.diffusion_steps,
        obs_history_len=args.obs_history_len,
        pred_horizon=args.pred_horizon,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    print(f"\n  Model Configuration:")
    print(f"    d_model: {config.d_model}")
    print(f"    n_heads: {config.n_heads}")
    print(f"    n_layers: {config.n_layers}")
    print(f"    d_ff: {config.d_ff}")
    print(f"    diffusion_steps: {config.diffusion_steps}")
    print(f"    obs_history_len: {config.obs_history_len}")
    print(f"    pred_horizon: {config.pred_horizon}")
    
    # 创建模型
    model = TransformerObstacleDiffusionWrapper(config, device)
    num_params = sum(p.numel() for p in model.model.parameters())
    print(f"    Total parameters: {num_params:,}")
    
    # 生成训练数据
    print(f"\n  Generating {args.num_train} training samples...")
    train_hist, train_fut = generate_complex_training_data(args.num_train, device=device)
    
    print(f"  Generating {args.num_test} test samples...")
    test_hist, test_fut = generate_complex_training_data(args.num_test, device=device)
    
    # 训练
    print(f"\n  Training for {args.num_epochs} epochs...")
    best_loss = float('inf')
    best_ade = float('inf')
    
    for epoch in tqdm(range(args.num_epochs), desc="  Training"):
        loss = model.train_epoch(train_hist, train_fut)
        
        # 定期评估
        if (epoch + 1) % args.eval_interval == 0:
            # 计算测试ADE
            errors = []
            for i in range(min(50, args.num_test)):
                mean_pred, _, _ = model.predict(test_hist[i], num_samples=10)
                ade = np.linalg.norm(mean_pred.numpy() - test_fut[i].cpu().numpy(), axis=-1).mean()
                errors.append(ade)
            avg_ade = np.mean(errors)
            
            tqdm.write(f"    Epoch {epoch+1:4d}: loss={loss:.4f}, test_ADE={avg_ade:.4f}")
            
            # 保存最佳模型
            if avg_ade < best_ade:
                best_ade = avg_ade
                best_loss = loss
                
                # 保存最佳模型
                best_model_path = os.path.join(args.save_dir, 'best_transformer_obstacle_model.pt')
                model.save(best_model_path)
                tqdm.write(f"    [NEW BEST] Saved to {best_model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'final_transformer_obstacle_model.pt')
    model.save(final_model_path)
    
    # 保存带时间戳的版本
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = os.path.join(args.save_dir, f'transformer_obstacle_model_{timestamp}.pt')
    model.save(timestamped_path)
    
    print("\n" + "=" * 70)
    print("  Training Complete!")
    print("=" * 70)
    print(f"\n  Results:")
    print(f"    Final Loss: {loss:.4f}")
    print(f"    Best ADE: {best_ade:.4f}")
    print(f"\n  Saved Models:")
    print(f"    Best:  {best_model_path}")
    print(f"    Final: {final_model_path}")
    print(f"    Timestamped: {timestamped_path}")
    
    print("\n  Usage in simulation:")
    print("    from src.transformer_obstacle_diffusion import TransformerObstacleDiffusionWrapper")
    print("    model = TransformerObstacleDiffusionWrapper.load_from_checkpoint(")
    print(f"        '{best_model_path}', device='cuda')")
    print("    prediction = model.predict(obstacle_history)")
    
    return model, best_ade


def main():
    parser = argparse.ArgumentParser(description='Train Upper Diffusion Model for Simulation')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--num_train', type=int, default=10000, help='训练样本数')
    parser.add_argument('--num_test', type=int, default=500, help='测试样本数')
    parser.add_argument('--eval_interval', type=int, default=50, help='评估间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cuda/cpu)')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='Transformer维度')
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--d_ff', type=int, default=512, help='FFN维度')
    parser.add_argument('--diffusion_steps', type=int, default=100, help='扩散步数')
    parser.add_argument('--obs_history_len', type=int, default=8, help='历史轨迹长度')
    parser.add_argument('--pred_horizon', type=int, default=12, help='预测时域')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    
    # 保存路径
    parser.add_argument('--save_dir', type=str, default=None, help='模型保存目录')
    
    args = parser.parse_args()
    
    # 设置保存目录
    if args.save_dir is None:
        args.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'trained_models'
        )
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练
    train_and_save_model(args)


if __name__ == '__main__':
    main()

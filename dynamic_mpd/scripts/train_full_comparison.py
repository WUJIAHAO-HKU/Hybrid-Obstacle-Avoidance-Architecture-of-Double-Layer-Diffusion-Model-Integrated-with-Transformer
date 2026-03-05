#!/usr/bin/env python3
"""
正式训练对比实验 - Transformer vs MLP
使用完整的复杂障碍物数据集进行训练

训练配置:
- 数据量: 10000 训练 + 2000 测试
- Epochs: 200
- 5种运动模式: circular, linear, zigzag, spiral, acceleration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime

# 固定随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("Loading models...")

from src.transformer_obstacle_diffusion import (
    TransformerObstacleDiffusionWrapper, 
    TransformerDiffusionConfig
)
from src.trainable_obstacle_diffusion import (
    TrainableObstacleDiffusion,
    DiffusionConfig
)
from src.complex_obstacle_data import (
    ComplexObstacleDataGenerator,
    ObstacleMotionConfig
)


def generate_full_dataset(n_train=10000, n_test=2000):
    """生成完整的训练和测试数据集"""
    motion_types = ['circular', 'linear', 'zigzag', 'spiral', 'acceleration']
    
    config = ObstacleMotionConfig(
        arena_size=2.0,
        obs_history_len=8,
        pred_horizon=12
    )
    generator = ComplexObstacleDataGenerator(config)
    
    def generate_samples(n_samples):
        all_history = []
        all_future = []
        all_types = []
        
        samples_per_type = n_samples // len(motion_types)
        
        for motion_type in motion_types:
            for _ in range(samples_per_type):
                traj = generator.generate_single_obstacle_trajectory(motion_type, total_steps=25)
                history = traj[:config.obs_history_len]
                future = traj[config.obs_history_len:config.obs_history_len + config.pred_horizon]
                
                if len(future) == config.pred_horizon:
                    all_history.append(torch.tensor(history, dtype=torch.float32))
                    all_future.append(torch.tensor(future, dtype=torch.float32))
                    all_types.append(motion_type)
        
        return torch.stack(all_history), torch.stack(all_future), all_types
    
    print(f"  Generating {n_train} training samples...")
    train_h, train_f, train_types = generate_samples(n_train)
    
    print(f"  Generating {n_test} test samples...")
    test_h, test_f, test_types = generate_samples(n_test)
    
    return train_h, train_f, test_h, test_f, train_types, test_types


def evaluate_model(model, test_history, test_future, device, num_eval=500):
    """评估模型在测试集上的表现"""
    errors_by_type = {'all': []}
    
    with torch.no_grad():
        indices = np.random.choice(len(test_history), min(num_eval, len(test_history)), replace=False)
        
        for i in indices:
            h = test_history[i].to(device)
            gt = test_future[i].cpu().numpy()
            
            pred_mean, _, _ = model.predict(h, num_samples=10)
            pred = pred_mean.cpu().numpy() if pred_mean.is_cuda else pred_mean.numpy()
            
            # ADE (Average Displacement Error)
            ade = np.mean(np.linalg.norm(pred - gt, axis=1))
            # FDE (Final Displacement Error)
            fde = np.linalg.norm(pred[-1] - gt[-1])
            
            errors_by_type['all'].append({'ade': ade, 'fde': fde})
    
    ade_mean = np.mean([e['ade'] for e in errors_by_type['all']])
    fde_mean = np.mean([e['fde'] for e in errors_by_type['all']])
    
    return ade_mean, fde_mean


def train_model(model, model_name, train_h, train_f, test_h, test_f, 
                num_epochs=200, batch_size=64, device='cuda', eval_every=20):
    """训练模型"""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Epochs: {num_epochs}, Batch Size: {batch_size}")
    print(f"{'='*60}")
    
    n_samples = len(train_h)
    history = {
        'train_loss': [],
        'test_ade': [],
        'test_fde': [],
        'epochs': []
    }
    
    best_ade = float('inf')
    
    pbar = tqdm(range(num_epochs), desc=model_name)
    
    for epoch in pbar:
        # 训练
        epoch_losses = []
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            h_batch = train_h[batch_idx].to(device)
            f_batch = train_f[batch_idx].to(device)
            
            if hasattr(model, 'trainer'):
                loss = model.trainer.train_step(h_batch, f_batch)
            else:
                loss = model.train_step(h_batch, f_batch)
            
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        
        # 定期评估
        if (epoch + 1) % eval_every == 0 or epoch == 0 or epoch == num_epochs - 1:
            if hasattr(model, 'model'):
                model.model.eval()
            
            ade, fde = evaluate_model(model, test_h, test_f, device, num_eval=300)
            history['test_ade'].append(ade)
            history['test_fde'].append(fde)
            history['epochs'].append(epoch)
            
            if ade < best_ade:
                best_ade = ade
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ADE': f'{ade:.4f}',
                'FDE': f'{fde:.4f}',
                'best': f'{best_ade:.4f}'
            })
    
    return history, best_ade


def visualize_predictions(model_mlp, model_trans, test_h, test_f, test_types, 
                          device, save_dir):
    """可视化预测结果 - 每种运动类型各一个"""
    os.makedirs(save_dir, exist_ok=True)
    
    motion_types = ['circular', 'linear', 'zigzag', 'spiral', 'acceleration']
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for col, motion_type in enumerate(motion_types):
        # 找到该类型的样本
        type_indices = [i for i, t in enumerate(test_types) if t == motion_type]
        if not type_indices:
            continue
        
        idx = type_indices[0]
        
        history = test_h[idx].cpu().numpy()
        gt_future = test_f[idx].cpu().numpy()
        h_tensor = test_h[idx].to(device)
        
        # 预测
        with torch.no_grad():
            mlp_pred, _, _ = model_mlp.predict(h_tensor, num_samples=15)
            trans_pred, _, _ = model_trans.predict(h_tensor, num_samples=15)
        
        mlp_pred = mlp_pred.cpu().numpy() if mlp_pred.is_cuda else mlp_pred.numpy()
        trans_pred = trans_pred.cpu().numpy() if trans_pred.is_cuda else trans_pred.numpy()
        
        # MLP预测
        ax = axes[0, col]
        ax.plot(history[:, 0], history[:, 1], 'ko-', markersize=5, label='History', linewidth=2)
        ax.plot(gt_future[:, 0], gt_future[:, 1], 'g-s', markersize=5, label='GT', linewidth=2)
        ax.plot(mlp_pred[:, 0], mlp_pred[:, 1], 'b--^', markersize=4, label='MLP', linewidth=2, alpha=0.8)
        
        mlp_ade = np.mean(np.linalg.norm(mlp_pred - gt_future, axis=1))
        ax.set_title(f'{motion_type.upper()}\nMLP ADE: {mlp_ade:.3f}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('MLP Baseline', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Transformer预测
        ax = axes[1, col]
        ax.plot(history[:, 0], history[:, 1], 'ko-', markersize=5, label='History', linewidth=2)
        ax.plot(gt_future[:, 0], gt_future[:, 1], 'g-s', markersize=5, label='GT', linewidth=2)
        ax.plot(trans_pred[:, 0], trans_pred[:, 1], 'r--v', markersize=4, label='Transformer', linewidth=2, alpha=0.8)
        
        trans_ade = np.mean(np.linalg.norm(trans_pred - gt_future, axis=1))
        ax.set_title(f'Transformer ADE: {trans_ade:.3f}', fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Transformer (Ours)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Trajectory Prediction Comparison: MLP vs Transformer\n'
                 'Row 1: MLP Baseline | Row 2: Transformer (Ours)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_by_motion_type.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(save_dir, 'prediction_by_motion_type.png')}")


def plot_training_curves(history_mlp, history_trans, save_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 训练损失
    ax = axes[0]
    ax.plot(history_mlp['train_loss'], 'b-', label='MLP', linewidth=2, alpha=0.7)
    ax.plot(history_trans['train_loss'], 'r-', label='Transformer', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ADE
    ax = axes[1]
    ax.plot(history_mlp['epochs'], history_mlp['test_ade'], 'b-o', label='MLP', linewidth=2)
    ax.plot(history_trans['epochs'], history_trans['test_ade'], 'r-s', label='Transformer', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('ADE (Average Displacement Error)', fontsize=11)
    ax.set_title('Test ADE Over Training', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FDE
    ax = axes[2]
    ax.plot(history_mlp['epochs'], history_mlp['test_fde'], 'b-o', label='MLP', linewidth=2)
    ax.plot(history_trans['epochs'], history_trans['test_fde'], 'r-s', label='Transformer', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('FDE (Final Displacement Error)', fontsize=11)
    ax.set_title('Test FDE Over Training', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(save_dir, 'training_curves.png')}")


def create_results_table(history_mlp, history_trans, best_mlp, best_trans, save_dir):
    """创建结果对比表"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # 计算最终指标
    final_mlp_ade = history_mlp['test_ade'][-1]
    final_mlp_fde = history_mlp['test_fde'][-1]
    final_trans_ade = history_trans['test_ade'][-1]
    final_trans_fde = history_trans['test_fde'][-1]
    
    improvement_ade = (final_mlp_ade - final_trans_ade) / final_mlp_ade * 100
    improvement_fde = (final_mlp_fde - final_trans_fde) / final_mlp_fde * 100
    
    data = [
        ['Metric', 'MLP Baseline', 'Transformer (Ours)', 'Improvement'],
        ['Final ADE', f'{final_mlp_ade:.4f}', f'{final_trans_ade:.4f}', f'{improvement_ade:.1f}%'],
        ['Final FDE', f'{final_mlp_fde:.4f}', f'{final_trans_fde:.4f}', f'{improvement_fde:.1f}%'],
        ['Best ADE', f'{best_mlp:.4f}', f'{best_trans:.4f}', f'{(best_mlp-best_trans)/best_mlp*100:.1f}%'],
        ['Final Loss', f'{history_mlp["train_loss"][-1]:.4f}', f'{history_trans["train_loss"][-1]:.4f}', '-'],
    ]
    
    colors = [['lightgray']*4]
    for row in data[1:]:
        colors.append(['white', 'lightyellow', 'lightgreen', 'lightcyan'])
    
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     cellColours=colors)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    ax.set_title('Quantitative Results: Transformer vs MLP\n'
                 f'ADE Improvement: {improvement_ade:.1f}% | FDE Improvement: {improvement_fde:.1f}%',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(save_dir, 'results_table.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(save_dir, 'results_table.png')}")
    
    return improvement_ade, improvement_fde


def main():
    print("\n" + "="*60)
    print("  Full Training: Transformer vs MLP Comparison")
    print("  Target: Top Conference (CoRL/ICRA/IROS)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    # 配置
    N_TRAIN = 10000
    N_TEST = 2000
    NUM_EPOCHS = 200
    BATCH_SIZE = 64
    EVAL_EVERY = 20
    
    # 生成数据
    print(f"\n  Generating dataset...")
    train_h, train_f, test_h, test_f, train_types, test_types = generate_full_dataset(N_TRAIN, N_TEST)
    print(f"  Train samples: {len(train_h)}")
    print(f"  Test samples: {len(test_h)}")
    
    # 创建模型
    print(f"\n  Creating models...")
    
    # MLP baseline
    mlp_config = DiffusionConfig(
        obs_history_len=8,
        pred_horizon=12,
        hidden_dim=256,
        diffusion_steps=50
    )
    model_mlp = TrainableObstacleDiffusion(mlp_config, device=device)
    
    # Transformer (ours)
    trans_config = TransformerDiffusionConfig(
        obs_history_len=8,
        pred_horizon=12,
        state_dim=2,
        d_model=128,
        n_heads=4,
        n_layers=4,
        diffusion_steps=50
    )
    model_trans = TransformerObstacleDiffusionWrapper(trans_config, device=device)
    
    # 统计参数量
    mlp_params = sum(p.numel() for p in model_mlp.denoise_net.parameters())
    trans_params = sum(p.numel() for p in model_trans.model.parameters())
    print(f"  MLP parameters: {mlp_params:,}")
    print(f"  Transformer parameters: {trans_params:,}")
    
    # 训练
    history_mlp, best_mlp = train_model(
        model_mlp, "MLP Baseline", train_h, train_f, test_h, test_f,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=device, eval_every=EVAL_EVERY
    )
    
    history_trans, best_trans = train_model(
        model_trans, "Transformer (Ours)", train_h, train_f, test_h, test_f,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=device, eval_every=EVAL_EVERY
    )
    
    # 可视化
    print(f"\n  Generating visualizations...")
    save_dir = 'results/full_comparison'
    os.makedirs(save_dir, exist_ok=True)
    
    visualize_predictions(model_mlp, model_trans, test_h, test_f, test_types, device, save_dir)
    plot_training_curves(history_mlp, history_trans, save_dir)
    improvement_ade, improvement_fde = create_results_table(
        history_mlp, history_trans, best_mlp, best_trans, save_dir
    )
    
    # 最终结果
    print(f"\n" + "="*60)
    print(f"  Final Results")
    print(f"="*60)
    print(f"  MLP Baseline:")
    print(f"    - Final ADE: {history_mlp['test_ade'][-1]:.4f}")
    print(f"    - Final FDE: {history_mlp['test_fde'][-1]:.4f}")
    print(f"    - Best ADE: {best_mlp:.4f}")
    print(f"\n  Transformer (Ours):")
    print(f"    - Final ADE: {history_trans['test_ade'][-1]:.4f}")
    print(f"    - Final FDE: {history_trans['test_fde'][-1]:.4f}")
    print(f"    - Best ADE: {best_trans:.4f}")
    print(f"\n  Improvement:")
    print(f"    - ADE: {improvement_ade:.1f}%")
    print(f"    - FDE: {improvement_fde:.1f}%")
    print(f"\n  Results saved to: {save_dir}/")
    print(f"="*60)
    
    # 保存模型
    model_mlp.save(os.path.join(save_dir, 'mlp_model.pt'))
    model_trans.save(os.path.join(save_dir, 'transformer_model.pt'))
    print(f"  Models saved!")


if __name__ == '__main__':
    main()

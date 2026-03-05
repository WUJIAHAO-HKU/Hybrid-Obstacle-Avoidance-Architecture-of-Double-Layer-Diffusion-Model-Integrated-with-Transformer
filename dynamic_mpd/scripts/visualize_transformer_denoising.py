"""
Transformer上层扩散模型 - 去噪过程可视化

展示从纯噪声(T=max)到清晰预测(T=0)的完整去噪过程
用于论文Figure展示

Author: Dynamic MPD Project
Date: 2026-01-30
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from tqdm import tqdm

from src.transformer_obstacle_diffusion import (
    TransformerObstacleDiffusionWrapper,
    TransformerDiffusionConfig,
    TransformerObstacleDiffusion
)
from src.complex_obstacle_data import (
    ComplexObstacleDataGenerator,
    ObstacleMotionConfig,
    generate_complex_training_data
)

# 设置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_or_train_model(device='cuda', model_path=None, force_train=False, num_epochs=300, num_train=8000):
    """
    加载预训练模型或训练新模型
    
    Args:
        device: 运行设备
        model_path: 预训练模型路径，None则使用默认路径
        force_train: 强制重新训练
        num_epochs: 训练轮数（仅在训练时使用）
        num_train: 训练样本数（仅在训练时使用）
    """
    # 默认模型路径
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'trained_models', 'best_transformer_obstacle_model.pt'
        )
    
    # 尝试加载预训练模型
    if os.path.exists(model_path) and not force_train:
        print("\n[1/3] Loading pretrained Transformer model...")
        print(f"  Model path: {model_path}")
        
        model = TransformerObstacleDiffusionWrapper.load_from_checkpoint(model_path, device)
        config = model.config
        
        # 生成测试数据用于可视化
        print("  Generating test samples for visualization...")
        test_hist, test_fut = generate_complex_training_data(500, device=device)
        
        return model, config, test_hist, test_fut
    
    # 如果没有预训练模型，则训练
    print("\n[1/3] Training Transformer Diffusion Model...")
    print(f"  (No pretrained model found at {model_path})")
    
    config = TransformerDiffusionConfig(
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        diffusion_steps=100,
        obs_history_len=8,
        pred_horizon=12,
        learning_rate=1e-4,
        batch_size=64
    )
    
    model = TransformerObstacleDiffusionWrapper(config, device)
    
    # 生成训练数据
    print(f"  Generating {num_train} training samples...")
    train_hist, train_fut = generate_complex_training_data(num_train, device=device)
    
    # 训练
    print(f"  Training for {num_epochs} epochs...")
    best_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc="  Training"):
        loss = model.train_epoch(train_hist, train_fut)
        if loss < best_loss:
            best_loss = loss
        if (epoch + 1) % 100 == 0:
            tqdm.write(f"    Epoch {epoch+1}: loss = {loss:.4f}")
    
    print(f"  Final loss: {loss:.4f}, Best loss: {best_loss:.4f}")
    
    # 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    # 生成测试数据
    test_hist, test_fut = generate_complex_training_data(500, device=device)
    
    return model, config, test_hist, test_fut


def visualize_denoising_process(model, config, test_hist, test_fut, device='cuda'):
    """
    可视化去噪过程
    
    类似于MLP版本的图，展示:
    - 4-6个不同样本
    - 每个样本展示: Target + 5个去噪阶段
    """
    print("\n[2/3] Creating denoising visualization...")
    
    model.model.eval()
    
    # 选择不同类型的样本 (3行)
    sample_indices = [0, 50, 100]
    num_samples_per_vis = 30  # 每个可视化生成的采样数
    
    # 去噪阶段 (根据扩散步数调整)
    T = config.diffusion_steps
    stages = [int(T*0.8), int(T*0.6), int(T*0.4), int(T*0.2), 0]
    
    fig = plt.figure(figsize=(26, 11))
    gs = GridSpec(len(sample_indices), 6, figure=fig, hspace=0.45, wspace=0.30)
    
    for row, sample_idx in enumerate(sample_indices):
        hist = test_hist[sample_idx:sample_idx+1].to(device)
        fut = test_fut[sample_idx].cpu().numpy()
        
        # 获取锚点（历史末端）用于相对位移
        anchor = hist[:, -1:, :]  # [1, 1, 2]
        rel_hist = hist - anchor
        
        with torch.no_grad():
            # 初始化纯噪声 (在相对坐标空间)
            x = torch.randn(num_samples_per_vis, config.pred_horizon, config.state_dim, device=device)
            recorded_states = {}
            
            # 扩展历史用于批量预测并编码 (使用相对坐标)
            rel_hist_expanded = rel_hist.expand(num_samples_per_vis, -1, -1)
            context = model.model.encoder(rel_hist_expanded)
            
            # 去噪过程 - 使用模型内置的扩散参数
            for step in reversed(range(T)):
                t = torch.full((num_samples_per_vis,), step, device=device, dtype=torch.long)
                
                # 使用Transformer decoder预测噪声
                noise_pred = model.model.decoder(x, t, context)
                
                # DDPM更新 - 使用模型的扩散参数
                alpha = model.model.alphas[step]
                alpha_bar = model.model.alphas_cumprod[step]
                alpha_bar_prev = model.model.alphas_cumprod_prev[step]
                beta = model.model.betas[step]
                
                # 计算x0预测
                x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
                x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
                
                # 计算后验均值
                posterior_mean = (
                    torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar) * x0_pred +
                    torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar) * x
                )
                
                # 添加噪声 (除了最后一步)
                if step > 0:
                    noise = torch.randn_like(x)
                    posterior_var = model.model.posterior_variance[step]
                    x = posterior_mean + torch.sqrt(posterior_var) * noise
                else:
                    x = posterior_mean
                
                # 记录特定阶段 - 转换回绝对坐标
                if step in stages:
                    abs_x = x + anchor.squeeze(0)  # 加回锚点
                    recorded_states[step] = abs_x.clone().cpu().numpy()
        
        hist_np = hist.squeeze(0).cpu().numpy()
        
        # ========== 第一列: 目标场景 ==========
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(hist_np[:, 0], hist_np[:, 1], 'g.-', linewidth=2.5, markersize=8, label='History')
        ax.plot(fut[:, 0], fut[:, 1], 'r.-', linewidth=2.5, markersize=8, label='Ground Truth')
        ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=120, marker='o', zorder=10, edgecolors='white', linewidth=2)
        ax.set_title(f'Sample {row+1}: Target', fontsize=16, fontweight='bold')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=13)
        if row == 0:
            ax.legend(fontsize=13, loc='upper right')
        
        # ========== 去噪阶段列 ==========
        for col, stage in enumerate(stages):
            ax = fig.add_subplot(gs[row, col + 1])
            
            states = recorded_states[stage]
            
            # 历史轨迹
            ax.plot(hist_np[:, 0], hist_np[:, 1], 'g-', linewidth=2, alpha=0.8)
            ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=100, zorder=10, edgecolors='white', linewidth=1.5)
            
            # 真实未来轨迹
            ax.plot(fut[:, 0], fut[:, 1], 'r--', linewidth=2, alpha=0.6)
            
            # 采样轨迹 (蓝色细线)
            for s in range(min(15, num_samples_per_vis)):
                ax.plot(states[s, :, 0], states[s, :, 1], 'b-', linewidth=0.6, alpha=0.25)
            
            # 均值预测 (蓝色粗线)
            mean_traj = states.mean(axis=0)
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=2.5, label='Mean Pred')
            
            # 计算误差
            error = np.linalg.norm(mean_traj - fut, axis=-1).mean()
            
            # 颜色编码误差
            if stage == 0:
                color = 'green' if error < 0.15 else 'orange' if error < 0.3 else 'red'
            else:
                color = 'black'
            
            ax.set_title(f'T={stage} | Err={error:.2f}', fontsize=15, fontweight='bold', color=color)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=13)
    
    fig.suptitle('Transformer Diffusion Denoising Process: From Noise (T=80) to Prediction (T=0)\n'
                 'Blue lines: sampled trajectories | Blue bold: mean prediction | Red dashed: ground truth',
                 fontsize=20, fontweight='bold', y=0.99)
    
    # 保存
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'transformer_denoising')
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, 'transformer_denoising_stages.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Saved: {output_path}")
    return output_path


def visualize_convergence_analysis(model, config, test_hist, test_fut, device='cuda'):
    """
    可视化收敛分析
    - 方差随时间变化
    - 误差随时间变化
    - 轨迹多样性演变
    """
    print("\n[3/3] Creating convergence analysis...")
    
    model.model.eval()
    
    T = config.diffusion_steps
    num_samples = 50
    
    # 选择一个样本进行详细分析
    hist = test_hist[0:1].to(device)
    fut = test_fut[0].cpu().numpy()
    
    # 获取锚点用于相对位移
    anchor = hist[:, -1:, :]  # [1, 1, 2]
    anchor_np = anchor.squeeze().cpu().numpy()
    rel_hist = hist - anchor
    
    # 计算相对未来轨迹
    rel_fut = fut - anchor_np
    
    # 记录所有阶段
    record_steps = list(range(0, T, max(1, T // 20)))  # 记录约20个点
    if 0 not in record_steps:
        record_steps.append(0)
    record_steps = sorted(set(record_steps), reverse=True)
    
    variances = []
    errors = []
    step_list = []
    
    with torch.no_grad():
        x = torch.randn(num_samples, config.pred_horizon, config.state_dim, device=device)
        
        rel_hist_expanded = rel_hist.expand(num_samples, -1, -1)
        context = model.model.encoder(rel_hist_expanded)
        
        for step in reversed(range(T)):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            noise_pred = model.model.decoder(x, t, context)
            
            alpha = model.model.alphas[step]
            alpha_bar = model.model.alphas_cumprod[step]
            alpha_bar_prev = model.model.alphas_cumprod_prev[step]
            beta = model.model.betas[step]
            
            x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
            posterior_mean = (
                torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar) * x0_pred +
                torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar) * x
            )
            
            if step > 0:
                noise = torch.randn_like(x)
                posterior_var = model.model.posterior_variance[step]
                x = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                x = posterior_mean
            
            if step in record_steps:
                x_np = x.cpu().numpy()
                var = np.mean(np.var(x_np, axis=0))
                mean_traj = x_np.mean(axis=0)
                # 使用相对坐标计算误差
                err = np.linalg.norm(mean_traj - rel_fut, axis=-1).mean()
                
                variances.append(var)
                errors.append(err)
                step_list.append(step)
    
    # 绘制分析图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. 方差收敛曲线
    ax = axes[0]
    ax.plot(step_list, variances, 'b-o', linewidth=3, markersize=10)
    ax.set_xlabel('Diffusion Step (t → 0)', fontsize=17)
    ax.set_ylabel('Mean Trajectory Variance', fontsize=17)
    ax.set_title('Variance Convergence', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # 2. 误差收敛曲线
    ax = axes[1]
    ax.plot(step_list, errors, 'r-o', linewidth=3, markersize=10)
    ax.set_xlabel('Diffusion Step (t → 0)', fontsize=17)
    ax.set_ylabel('Mean Prediction Error', fontsize=17)
    ax.set_title('Error Convergence', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # 标注最终误差
    final_err = errors[-1] if errors else 0
    ax.axhline(y=final_err, color='green', linestyle='--', alpha=0.7)
    ax.text(0.02, 0.95, f'Final Error: {final_err:.4f}', 
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            color='green', va='top')
    
    # 3. 初始 vs 最终轨迹对比
    ax = axes[2]
    hist_np = hist.squeeze(0).cpu().numpy()
    
    # 历史
    ax.plot(hist_np[:, 0], hist_np[:, 1], 'g.-', linewidth=2, markersize=8, label='History')
    ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=120, marker='o', zorder=10, edgecolors='white')
    
    # Ground Truth (绝对坐标)
    ax.plot(fut[:, 0], fut[:, 1], 'r--', linewidth=2.5, label='Ground Truth')
    
    # 最终预测 (使用相对位移)
    with torch.no_grad():
        x_final = torch.randn(20, config.pred_horizon, config.state_dim, device=device)
        rel_hist_exp = rel_hist.expand(20, -1, -1)
        context_final = model.model.encoder(rel_hist_exp)
        
        for step in reversed(range(T)):
            t = torch.full((20,), step, device=device, dtype=torch.long)
            noise_pred = model.model.decoder(x_final, t, context_final)
            alpha = model.model.alphas[step]
            alpha_bar = model.model.alphas_cumprod[step]
            alpha_bar_prev = model.model.alphas_cumprod_prev[step]
            beta = model.model.betas[step]
            x0_pred = (x_final - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
            posterior_mean = (
                torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar) * x0_pred +
                torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar) * x_final
            )
            if step > 0:
                noise = torch.randn_like(x_final)
                posterior_var = model.model.posterior_variance[step]
                x_final = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                x_final = posterior_mean
        
        # 转换回绝对坐标
        x_final_abs = x_final + anchor.squeeze(0)
        x_final_np = x_final_abs.cpu().numpy()
        for i in range(5):
            ax.plot(x_final_np[i, :, 0], x_final_np[i, :, 1], 'b-', linewidth=0.8, alpha=0.3)
        mean_final = x_final_np.mean(axis=0)
        ax.plot(mean_final[:, 0], mean_final[:, 1], 'b-', linewidth=2.5, label='Prediction')
    
    ax.set_title('Final Prediction vs Ground Truth', fontsize=18, fontweight='bold')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=15)
    
    fig.suptitle('Transformer Diffusion Model: Convergence Analysis', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'transformer_denoising')
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, 'transformer_convergence_analysis.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Saved: {output_path}")
    return output_path

def main():
    print("=" * 70)
    print("  Transformer Obstacle Diffusion - Denoising Process Visualization")
    print("  For Paper Figure")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    # 1. 加载预训练模型（如果存在），否则训练新模型
    model, config, test_hist, test_fut = load_or_train_model(
        device=device,
        model_path=None,    # 使用默认路径: trained_models/best_transformer_obstacle_model.pt
        force_train=False,  # 设为True可强制重新训练
    )
    
    # 2. 去噪过程可视化
    path1 = visualize_denoising_process(model, config, test_hist, test_fut, device)
    
    # 3. 收敛分析
    path2 = visualize_convergence_analysis(model, config, test_hist, test_fut, device)
    
    print("\n" + "=" * 70)
    print("  Visualization Complete!")
    print("=" * 70)
    print(f"\n  Output files:")
    print(f"    1. {path1}")
    print(f"    2. {path2}")
    print("\n  Use these figures in your paper!")


if __name__ == '__main__':
    main()

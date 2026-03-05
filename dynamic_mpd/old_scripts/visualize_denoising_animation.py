"""
上层扩散模型去噪过程动画 - 详细展示从纯噪声到预测的过程

展示:
1. 完整去噪动画 (每一步)
2. 分布变化过程
3. 训练效果对比

Author: Dynamic MPD Project
"""

import sys
sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from src.trainable_obstacle_diffusion import (
    TrainableObstacleDiffusion, DiffusionConfig, generate_training_data
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_model_thoroughly():
    """充分训练模型以确保效果达标"""
    
    print("=" * 60)
    print("  Training Upper-Level Diffusion Model (Extended)")
    print("=" * 60)
    
    device = 'cpu'
    
    config = DiffusionConfig(
        obs_history_len=8,
        pred_horizon=12,
        hidden_dim=256,       # 增大网络
        num_layers=4,         # 更深
        diffusion_steps=100,  # 更多步数
        learning_rate=5e-4,
        batch_size=128
    )
    
    model = TrainableObstacleDiffusion(config, device)
    
    # 生成更多数据
    print("\n[1/3] Generating training data (5000 samples)...")
    num_data = 5000
    histories, futures = generate_training_data(num_data, obs_len=8, pred_len=12, device=device)
    
    train_hist, test_hist = histories[:4000], histories[4000:]
    train_fut, test_fut = futures[:4000], futures[4000:]
    
    # 更充分的训练
    print("\n[2/3] Training model (200 epochs)...")
    
    loss_history = []
    num_epochs = 200
    batch_size = config.batch_size
    
    for epoch in range(num_epochs):
        perm = torch.randperm(len(train_hist))
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_hist), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 2:
                continue
            loss = model.train_step(train_hist[idx], train_fut[idx])
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}: Loss={avg_loss:.4f}")
    
    print(f"  [OK] Final Loss: {loss_history[-1]:.4f}")
    
    return model, config, test_hist, test_fut, loss_history


def create_denoising_animation(model, config, test_hist, test_fut):
    """创建去噪过程动画"""
    
    print("\n[3/3] Creating denoising animation...")
    
    device = 'cpu'
    model.denoise_net.eval()
    
    # 选择5个测试样本
    sample_indices = [0, 50, 100, 150, 200]
    num_parallel = 20  # 每个样本并行采样数
    
    # 记录所有去噪步骤
    all_denoising_states = []  # [step, sample_idx, parallel_idx, horizon, 2]
    
    print("  Recording denoising process for 5 samples...")
    
    with torch.no_grad():
        for sample_idx in sample_indices:
            hist = test_hist[sample_idx:sample_idx+1]  # [1, 8, 2]
            
            # 初始化噪声
            x_batch = torch.randn(num_parallel, 12, 2, device=device)
            sample_states = [x_batch.clone()]
            
            for step in reversed(range(config.diffusion_steps)):
                t = torch.tensor([step], device=device)
                
                # 为每个并行样本预测
                noise_pred = model.denoise_net(
                    x_batch,
                    hist.expand(num_parallel, -1, -1),
                    t.expand(num_parallel)
                )
                
                alpha = model.alphas[step]
                alpha_bar = model.alpha_bars[step]
                
                x_batch = (1 / alpha.sqrt()) * (
                    x_batch - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred
                )
                
                if step > 0:
                    noise = torch.randn_like(x_batch) * model.betas[step].sqrt()
                    x_batch = x_batch + noise
                
                # 每隔几步记录
                if step % 5 == 0 or step < 5:
                    sample_states.append(x_batch.clone())
            
            all_denoising_states.append(sample_states)
    
    # 创建动画
    num_frames = len(all_denoising_states[0])
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Upper-Level Diffusion: Denoising Process\n'
                 'From Pure Noise (T=100) → Predicted Trajectory (T=0)',
                 fontsize=14, fontweight='bold')
    
    # 设置每个子图
    for ax in axes:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # 预计算真实轨迹和历史
    histories_np = [test_hist[idx].squeeze(0).numpy() for idx in sample_indices]
    futures_np = [test_fut[idx].numpy() for idx in sample_indices]
    
    lines = []
    scatter_plots = []
    ellipses = []
    titles = []
    
    for i, ax in enumerate(axes):
        # 历史轨迹 (固定)
        ax.plot(histories_np[i][:, 0], histories_np[i][:, 1], 'g.-', 
                linewidth=2, markersize=6, label='History')
        ax.scatter(histories_np[i][-1, 0], histories_np[i][-1, 1], 
                   c='green', s=100, marker='o', zorder=10)
        
        # 真实未来轨迹 (虚线参考)
        ax.plot(futures_np[i][:, 0], futures_np[i][:, 1], 'r--', 
                linewidth=2, alpha=0.5, label='Ground Truth')
        
        # 预测轨迹 (动态)
        line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.8, label='Mean Pred')
        lines.append(line)
        
        # 并行采样散点
        scatter = ax.scatter([], [], c='blue', s=8, alpha=0.3)
        scatter_plots.append(scatter)
        
        # 不确定性椭圆
        ellipse = Ellipse((0, 0), 0, 0, facecolor='blue', alpha=0.1, edgecolor='blue')
        ax.add_patch(ellipse)
        ellipses.append(ellipse)
        
        title = ax.set_title(f'Sample {i+1}', fontsize=11)
        titles.append(title)
        
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')
    
    def init():
        for line in lines:
            line.set_data([], [])
        for scatter in scatter_plots:
            scatter.set_offsets(np.empty((0, 2)))
        return lines + scatter_plots
    
    def update(frame):
        step_label = config.diffusion_steps - frame * 5 if frame < len(all_denoising_states[0])-1 else 0
        
        for i in range(5):
            states = all_denoising_states[i][frame]  # [num_parallel, 12, 2]
            states_np = states.numpy()
            
            # 均值轨迹
            mean_traj = states_np.mean(axis=0)  # [12, 2]
            lines[i].set_data(mean_traj[:, 0], mean_traj[:, 1])
            
            # 所有采样点 (最后一个时间步)
            end_points = states_np[:, -1, :]  # [num_parallel, 2]
            scatter_plots[i].set_offsets(end_points)
            
            # 不确定性椭圆 (基于最后时间步的分布)
            std = states_np[:, -1, :].std(axis=0)
            mean = states_np[:, -1, :].mean(axis=0)
            ellipses[i].set_center((mean[0], mean[1]))
            ellipses[i].set_width(std[0] * 4)
            ellipses[i].set_height(std[1] * 4)
            
            # 计算当前误差
            error = np.linalg.norm(mean_traj - futures_np[i], axis=-1).mean()
            titles[i].set_text(f'Sample {i+1} | T={step_label} | Err={error:.2f}')
        
        fig.suptitle(f'Upper-Level Diffusion: Denoising Process\n'
                     f'Step: T={step_label} → T=0 (Frame {frame+1}/{num_frames})',
                     fontsize=14, fontweight='bold')
        
        return lines + scatter_plots + ellipses + titles
    
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, 
                         blit=False, interval=200)
    
    output_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/diffusion_denoising.gif'
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()
    
    print(f"  [OK] Saved: {output_path}")
    
    return output_path


def create_static_comparison(model, config, test_hist, test_fut, loss_history):
    """创建静态对比图"""
    
    print("\n  Creating static comparison figure...")
    
    device = 'cpu'
    model.denoise_net.eval()
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25)
    
    # ----- 1. Loss曲线 -----
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(loss_history, 'b-', linewidth=2)
    ax_loss.set_xlabel('Epoch', fontsize=11)
    ax_loss.set_ylabel('Loss', fontsize=11)
    ax_loss.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.text(0.95, 0.95, f'Final: {loss_history[-1]:.4f}',
                 transform=ax_loss.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ----- 2-4. 去噪阶段展示 -----
    sample_idx = 50
    hist = test_hist[sample_idx:sample_idx+1]
    fut = test_fut[sample_idx]
    
    stages = [100, 50, 20, 0]
    stage_labels = ['T=100 (Noise)', 'T=50', 'T=20', 'T=0 (Pred)']
    num_samples = 30
    
    with torch.no_grad():
        x = torch.randn(num_samples, 12, 2, device=device)
        recorded_states = {100: x.clone()}
        
        for step in reversed(range(config.diffusion_steps)):
            t = torch.tensor([step], device=device).expand(num_samples)
            noise_pred = model.denoise_net(x, hist.expand(num_samples, -1, -1), t)
            
            alpha = model.alphas[step]
            alpha_bar = model.alpha_bars[step]
            
            x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred)
            
            if step > 0:
                x = x + torch.randn_like(x) * model.betas[step].sqrt()
            
            if step in stages:
                recorded_states[step] = x.clone()
    
    hist_np = hist.squeeze(0).numpy()
    fut_np = fut.numpy()
    
    for i, (stage, label) in enumerate(zip(stages, stage_labels)):
        ax = fig.add_subplot(gs[0, i])
        
        states = recorded_states[stage].numpy()  # [30, 12, 2]
        
        # 历史
        ax.plot(hist_np[:, 0], hist_np[:, 1], 'g-', linewidth=2, label='History')
        ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=100, zorder=10)
        
        # 真实
        ax.plot(fut_np[:, 0], fut_np[:, 1], 'r--', linewidth=2, label='GT')
        
        # 所有采样
        for s in range(min(10, num_samples)):
            ax.plot(states[s, :, 0], states[s, :, 1], 'b-', linewidth=0.5, alpha=0.3)
        
        # 均值
        mean_traj = states.mean(axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=2, label='Mean')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(label, fontsize=11, fontweight='bold')
        
        if i == 0:
            ax.legend(fontsize=8)
    
    # ----- 5-8. 多样本预测对比 -----
    test_indices = [0, 100, 200, 300]
    
    for i, idx in enumerate(test_indices):
        ax = fig.add_subplot(gs[1, i])
        
        hist = test_hist[idx:idx+1]
        fut = test_fut[idx]
        
        with torch.no_grad():
            samples = []
            for _ in range(20):
                x = torch.randn(1, 12, 2, device=device)
                for step in reversed(range(config.diffusion_steps)):
                    t = torch.tensor([step], device=device)
                    noise_pred = model.denoise_net(x, hist, t)
                    alpha = model.alphas[step]
                    alpha_bar = model.alpha_bars[step]
                    x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred)
                    if step > 0:
                        x = x + torch.randn_like(x) * model.betas[step].sqrt()
                samples.append(x.squeeze(0))
            
            samples_stack = torch.stack(samples)
            mean_pred = samples_stack.mean(dim=0).numpy()
            error = np.linalg.norm(mean_pred - fut.numpy(), axis=-1).mean()
        
        hist_np = hist.squeeze(0).numpy()
        fut_np = fut.numpy()
        
        ax.plot(hist_np[:, 0], hist_np[:, 1], 'g-', linewidth=2)
        ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=80, zorder=10)
        ax.plot(fut_np[:, 0], fut_np[:, 1], 'r--', linewidth=2, label='GT')
        ax.plot(mean_pred[:, 0], mean_pred[:, 1], 'b-', linewidth=2, label='Pred')
        
        # 采样分布
        for s in samples[:8]:
            s_np = s.numpy()
            ax.plot(s_np[:, 0], s_np[:, 1], 'b-', linewidth=0.5, alpha=0.2)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Test {i+1}: Error={error:.3f}', fontsize=11, fontweight='bold')
        
        if i == 0:
            ax.legend(fontsize=8)
    
    fig.suptitle('Upper-Level Diffusion Model: Training & Prediction Results\n'
                 'Top: Denoising Stages | Bottom: Multi-Sample Predictions',
                 fontsize=14, fontweight='bold', y=0.98)
    
    output_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/diffusion_overview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Saved: {output_path}")
    
    return output_path


def main():
    # 训练模型
    model, config, test_hist, test_fut, loss_history = train_model_thoroughly()
    
    # 创建去噪动画
    anim_path = create_denoising_animation(model, config, test_hist, test_fut)
    
    # 创建静态对比图
    static_path = create_static_comparison(model, config, test_hist, test_fut, loss_history)
    
    # 计算最终指标
    print("\n" + "=" * 60)
    print("  Final Evaluation")
    print("=" * 60)
    
    errors = []
    model.denoise_net.eval()
    
    with torch.no_grad():
        for i in range(50):
            idx = i * 20
            if idx >= len(test_hist):
                break
            
            hist = test_hist[idx:idx+1]
            fut = test_fut[idx]
            
            samples = []
            for _ in range(10):
                x = torch.randn(1, 12, 2)
                for step in reversed(range(config.diffusion_steps)):
                    t = torch.tensor([step])
                    noise_pred = model.denoise_net(x, hist, t)
                    alpha = model.alphas[step]
                    alpha_bar = model.alpha_bars[step]
                    x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred)
                    if step > 0:
                        x = x + torch.randn_like(x) * model.betas[step].sqrt()
                samples.append(x.squeeze(0))
            
            mean_pred = torch.stack(samples).mean(dim=0)
            error = torch.norm(mean_pred - fut, dim=-1).mean().item()
            errors.append(error)
    
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"  Final Training Loss: {loss_history[-1]:.4f}")
    print(f"  Avg Prediction Error: {avg_error:.4f} ± {std_error:.4f}")
    print(f"  Quality Check: {'PASS ✓' if avg_error < 0.5 else 'NEEDS IMPROVEMENT'}")
    print("=" * 60)
    
    print(f"\nOutputs:")
    print(f"  1. {anim_path}")
    print(f"  2. {static_path}")


if __name__ == '__main__':
    main()

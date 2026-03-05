"""
可视化上层扩散模型的障碍物预测过程

展示:
1. 训练过程 (Loss曲线)
2. 扩散去噪过程 (从噪声到预测)
3. 预测 vs 真实轨迹对比
4. 多模态采样分布

Author: Dynamic MPD Project
"""

import sys
sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.gridspec import GridSpec
from src.trainable_obstacle_diffusion import (
    TrainableObstacleDiffusion, DiffusionConfig, generate_training_data
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_training_and_prediction():
    """完整可视化训练和预测过程"""
    
    print("=" * 60)
    print("  Upper-Level Diffusion Model Visualization")
    print("=" * 60)
    
    device = 'cpu'
    
    # ==================== 配置 ====================
    config = DiffusionConfig(
        obs_history_len=8,
        pred_horizon=12,
        hidden_dim=128,
        num_layers=3,
        diffusion_steps=50,
        learning_rate=1e-3,
        batch_size=64
    )
    
    model = TrainableObstacleDiffusion(config, device)
    
    # ==================== 生成数据 ====================
    print("\n[1/4] Generating training data...")
    num_data = 3000
    histories, futures = generate_training_data(num_data, obs_len=8, pred_len=12, device=device)
    
    # 分割训练/测试
    train_hist, test_hist = histories[:2500], histories[2500:]
    train_fut, test_fut = futures[:2500], futures[2500:]
    
    print(f"  Train: {len(train_hist)}, Test: {len(test_hist)}")
    
    # ==================== 训练并记录Loss ====================
    print("\n[2/4] Training model (100 epochs)...")
    
    loss_history = []
    test_loss_history = []
    num_epochs = 100
    batch_size = config.batch_size
    
    for epoch in range(num_epochs):
        # 训练
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
        
        # 测试Loss
        model.denoise_net.eval()
        with torch.no_grad():
            t = torch.randint(0, config.diffusion_steps, (len(test_hist),), device=device)
            noisy, noise = model.q_sample(test_fut, t)
            noise_pred = model.denoise_net(noisy, test_hist, t)
            test_loss = torch.nn.functional.mse_loss(noise_pred, noise).item()
        test_loss_history.append(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}: Train={avg_loss:.4f}, Test={test_loss:.4f}")
    
    print("  [OK] Training complete!")
    
    # ==================== 创建可视化 ====================
    print("\n[3/4] Creating visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # ----- 1. Loss曲线 -----
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(loss_history, 'b-', linewidth=2, label='Train Loss')
    ax1.plot(test_loss_history, 'r--', linewidth=2, label='Test Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, num_epochs)
    
    # 添加收敛指标
    final_train = loss_history[-1]
    final_test = test_loss_history[-1]
    ax1.text(0.98, 0.95, f'Final Train: {final_train:.4f}\nFinal Test: {final_test:.4f}',
             transform=ax1.transAxes, ha='right', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ----- 2. 扩散去噪过程可视化 -----
    # 选择一个测试样本
    test_idx = 10
    test_history = test_hist[test_idx:test_idx+1]  # [1, 8, 2]
    test_future = test_fut[test_idx]               # [12, 2]
    
    # 记录去噪过程中的中间状态
    print("  Recording denoising process...")
    model.denoise_net.eval()
    
    with torch.no_grad():
        x = torch.randn(1, 12, 2, device=device)
        denoising_states = [x.squeeze(0).clone()]
        
        for step in reversed(range(config.diffusion_steps)):
            t = torch.tensor([step], device=device)
            noise_pred = model.denoise_net(x, test_history, t)
            
            alpha = model.alphas[step]
            alpha_bar = model.alpha_bars[step]
            
            x = (1 / alpha.sqrt()) * (
                x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred
            )
            
            if step > 0:
                noise = torch.randn_like(x) * model.betas[step].sqrt()
                x = x + noise
            
            # 记录关键步骤
            if step in [45, 35, 25, 15, 5, 0]:
                denoising_states.append(x.squeeze(0).clone())
    
    # 绘制去噪过程
    denoise_steps = ['T=50\n(Noise)', 'T=45', 'T=35', 'T=25', 'T=15', 'T=5', 'T=0\n(Pred)']
    
    for i, (state, title) in enumerate(zip(denoising_states, denoise_steps)):
        ax = fig.add_subplot(gs[0, 2]) if i == 0 else fig.add_subplot(gs[0, 3]) if i == len(denoising_states)-1 else None
        if ax is None:
            continue
            
        state_np = state.numpy()
        hist_np = test_history.squeeze(0).numpy()
        fut_np = test_future.numpy()
        
        # 历史轨迹
        ax.plot(hist_np[:, 0], hist_np[:, 1], 'g.-', linewidth=2, markersize=8, label='History')
        ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=100, marker='o', zorder=10)
        
        if i == 0:
            # 噪声
            ax.scatter(state_np[:, 0], state_np[:, 1], c='gray', s=30, alpha=0.5)
            ax.set_title('Initial: Pure Noise', fontsize=12, fontweight='bold')
        else:
            # 预测
            ax.plot(state_np[:, 0], state_np[:, 1], 'b.-', linewidth=2, markersize=6, label='Prediction')
            # 真实
            ax.plot(fut_np[:, 0], fut_np[:, 1], 'r--', linewidth=2, alpha=0.7, label='Ground Truth')
            ax.set_title('Final: Denoised Prediction', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # ----- 3. 去噪过程详细展示 (6个阶段) -----
    for i, (state, title) in enumerate(zip(denoising_states, denoise_steps)):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        state_np = state.numpy()
        hist_np = test_history.squeeze(0).numpy()
        fut_np = test_future.numpy()
        
        # 历史
        ax.plot(hist_np[:, 0], hist_np[:, 1], 'g-', linewidth=2, alpha=0.7)
        ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=80, marker='o', zorder=10)
        
        # 当前状态
        if i == 0:
            ax.scatter(state_np[:, 0], state_np[:, 1], c='purple', s=20, alpha=0.5)
        else:
            ax.plot(state_np[:, 0], state_np[:, 1], 'b-', linewidth=2, alpha=0.8)
            ax.scatter(state_np[:, 0], state_np[:, 1], c='blue', s=20)
        
        # 真实轨迹 (虚线参考)
        ax.plot(fut_np[:, 0], fut_np[:, 1], 'r--', linewidth=1.5, alpha=0.5)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # ----- 4. 多模态采样 (最后一个位置) -----
    ax_multi = fig.add_subplot(gs[2, 2:4])
    
    print("  Generating multi-modal samples...")
    num_samples = 30
    all_samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(1, 12, 2, device=device)
            
            for step in reversed(range(config.diffusion_steps)):
                t = torch.tensor([step], device=device)
                noise_pred = model.denoise_net(x, test_history, t)
                
                alpha = model.alphas[step]
                alpha_bar = model.alpha_bars[step]
                
                x = (1 / alpha.sqrt()) * (
                    x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred
                )
                
                if step > 0:
                    noise = torch.randn_like(x) * model.betas[step].sqrt()
                    x = x + noise
            
            all_samples.append(x.squeeze(0))
    
    # 绘制所有采样
    hist_np = test_history.squeeze(0).numpy()
    fut_np = test_future.numpy()
    
    ax_multi.plot(hist_np[:, 0], hist_np[:, 1], 'g-', linewidth=3, label='History')
    ax_multi.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=150, marker='o', zorder=10)
    
    for sample in all_samples:
        sample_np = sample.numpy()
        ax_multi.plot(sample_np[:, 0], sample_np[:, 1], 'b-', linewidth=0.8, alpha=0.3)
    
    # 均值
    samples_stack = torch.stack(all_samples)
    mean_pred = samples_stack.mean(dim=0).numpy()
    std_pred = samples_stack.std(dim=0).numpy()
    
    ax_multi.plot(mean_pred[:, 0], mean_pred[:, 1], 'b-', linewidth=3, label='Mean Prediction')
    ax_multi.plot(fut_np[:, 0], fut_np[:, 1], 'r--', linewidth=3, label='Ground Truth')
    
    # 置信椭圆
    for t in [0, 5, 11]:
        ellipse = Ellipse(
            (mean_pred[t, 0], mean_pred[t, 1]),
            width=std_pred[t, 0] * 4,
            height=std_pred[t, 1] * 4,
            facecolor='blue',
            alpha=0.15,
            edgecolor='blue',
            linewidth=2
        )
        ax_multi.add_patch(ellipse)
    
    ax_multi.set_title(f'Multi-Modal Sampling ({num_samples} samples)', fontsize=14, fontweight='bold')
    ax_multi.legend(fontsize=11, loc='upper left')
    ax_multi.set_xlim(-1.5, 1.5)
    ax_multi.set_ylim(-1.5, 1.5)
    ax_multi.set_aspect('equal')
    ax_multi.grid(True, alpha=0.3)
    
    # 计算预测误差
    pred_error = torch.norm(samples_stack.mean(dim=0) - test_future, dim=-1).mean().item()
    ax_multi.text(0.98, 0.02, f'Mean Prediction Error: {pred_error:.4f}',
                  transform=ax_multi.transAxes, ha='right', va='bottom', fontsize=11,
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 总标题
    fig.suptitle('Upper-Level Diffusion Model: Obstacle Trajectory Prediction\n'
                 'Training Process + Denoising Visualization + Multi-Modal Sampling',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 保存
    output_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/obstacle_diffusion_training.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n[OK] Saved: {output_path}")
    
    # ==================== 评估多个测试样本 ====================
    print("\n[4/4] Evaluating on test set...")
    
    fig2, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    test_errors = []
    
    for i in range(12):
        ax = axes[i // 4, i % 4]
        
        idx = i * 40  # 均匀采样测试集
        if idx >= len(test_hist):
            idx = i
        
        hist = test_hist[idx:idx+1]
        fut = test_fut[idx]
        
        # 预测
        with torch.no_grad():
            samples = []
            for _ in range(15):
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
            
            mean_pred = torch.stack(samples).mean(dim=0)
            error = torch.norm(mean_pred - fut, dim=-1).mean().item()
            test_errors.append(error)
        
        # 绘制
        hist_np = hist.squeeze(0).numpy()
        fut_np = fut.numpy()
        pred_np = mean_pred.numpy()
        
        ax.plot(hist_np[:, 0], hist_np[:, 1], 'g-', linewidth=2)
        ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=60, zorder=10)
        ax.plot(fut_np[:, 0], fut_np[:, 1], 'r--', linewidth=2, label='GT')
        ax.plot(pred_np[:, 0], pred_np[:, 1], 'b-', linewidth=2, label='Pred')
        
        ax.set_title(f'Test {i+1}: Error={error:.3f}', fontsize=10)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        if i == 0:
            ax.legend(fontsize=8)
    
    avg_error = np.mean(test_errors)
    fig2.suptitle(f'Test Set Evaluation (12 samples)\nAverage Prediction Error: {avg_error:.4f}',
                  fontsize=14, fontweight='bold')
    
    output_path2 = '/home/wujiahao/mpd-build/dynamic_mpd/results/obstacle_diffusion_evaluation.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[OK] Saved: {output_path2}")
    
    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Final Train Loss: {final_train:.4f}")
    print(f"  Final Test Loss:  {final_test:.4f}")
    print(f"  Avg Prediction Error: {avg_error:.4f}")
    print(f"  Overfitting Check: {'OK' if final_test < final_train * 1.5 else 'WARNING'}")
    print("=" * 60)
    
    return output_path, output_path2


if __name__ == '__main__':
    visualize_training_and_prediction()

"""
复杂障碍物扩散模型训练与可视化

特点:
1. 多种运动模式: 直线/弧线/变速/突然转向/螺旋等
2. 更多训练数据: 10000+ 样本
3. 更长训练: 300+ epochs
4. 详细去噪过程可视化

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
from src.trainable_obstacle_diffusion import TrainableObstacleDiffusion, DiffusionConfig
from src.complex_obstacle_data import (
    ComplexObstacleDataGenerator, 
    ObstacleMotionConfig,
    generate_complex_training_data
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_motion_types():
    """可视化各种运动类型"""
    print("\n[1/5] Visualizing motion types...")
    
    config = ObstacleMotionConfig(num_obstacles=1, obs_history_len=8, pred_horizon=12)
    gen = ComplexObstacleDataGenerator(config)
    
    motion_types = [
        'linear', 'arc', 'acceleration', 'deceleration',
        'variable_speed', 'zigzag', 'spiral', 'sudden_turn',
        'stop_and_go', 'curved', 'random_walk'
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    total_steps = config.obs_history_len + config.pred_horizon
    
    for i, mt in enumerate(motion_types):
        ax = axes[i]
        
        # 生成多条同类型轨迹
        for _ in range(5):
            traj = gen.generate_single_obstacle_trajectory(mt, total_steps)
            traj_np = traj.numpy()
            
            # 历史部分
            ax.plot(traj_np[:config.obs_history_len, 0], 
                    traj_np[:config.obs_history_len, 1],
                    'g-', linewidth=1.5, alpha=0.6)
            
            # 未来部分
            ax.plot(traj_np[config.obs_history_len-1:, 0], 
                    traj_np[config.obs_history_len-1:, 1],
                    'b-', linewidth=1.5, alpha=0.6)
            
            # 起点和终点
            ax.scatter(traj_np[0, 0], traj_np[0, 1], c='green', s=30, marker='o', alpha=0.5)
            ax.scatter(traj_np[-1, 0], traj_np[-1, 1], c='red', s=30, marker='x', alpha=0.5)
        
        ax.set_title(mt.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlim(-2.2, 2.2)  # 扩大显示范围
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # 最后一个子图显示图例
    ax = axes[11]
    ax.plot([], [], 'g-', linewidth=2, label='History (8 steps)')
    ax.plot([], [], 'b-', linewidth=2, label='Future (12 steps)')
    ax.scatter([], [], c='green', s=50, marker='o', label='Start')
    ax.scatter([], [], c='red', s=50, marker='x', label='End')
    ax.legend(loc='center', fontsize=12)
    ax.axis('off')
    ax.set_title('Legend', fontsize=11, fontweight='bold')
    
    fig.suptitle('Complex Obstacle Motion Types\n'
                 'Green: History | Blue: Future Prediction Target',
                 fontsize=14, fontweight='bold')
    
    output_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/motion_types.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Saved: {output_path}")
    return output_path


def train_with_complex_data():
    """使用复杂数据训练模型"""
    
    print("\n[2/5] Training with complex obstacle data...")
    
    device = 'cpu'
    
    # 更大的网络配置
    config = DiffusionConfig(
        obs_history_len=8,
        pred_horizon=12,
        hidden_dim=512,
        num_layers=6,
        diffusion_steps=150,
        learning_rate=1e-4,
        batch_size=128
    )
    
    model = TrainableObstacleDiffusion(config, device)
    
    # 生成大量复杂数据
    print("  Generating 24000 complex training samples...")
    num_train = 20000
    num_test = 4000
    
    train_hist, train_fut = generate_complex_training_data(num_train, device=device)
    test_hist, test_fut = generate_complex_training_data(num_test, device=device)
    
    print(f"  Train: {train_hist.shape}, Test: {test_hist.shape}")
    
    # 训练
    print("\n  Training for 1000 epochs...")
    
    loss_history = []
    test_loss_history = []
    num_epochs = 1000
    batch_size = config.batch_size
    
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.denoise_net.train()
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
            test_idx = torch.randperm(len(test_hist))[:500]
            t = torch.randint(0, config.diffusion_steps, (500,), device=device)
            noisy, noise = model.q_sample(test_fut[test_idx], t)
            noise_pred = model.denoise_net(noisy, test_hist[test_idx], t)
            test_loss = torch.nn.functional.mse_loss(noise_pred, noise).item()
        test_loss_history.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}: Train={avg_loss:.4f}, Test={test_loss:.4f}")
    
    print(f"  [OK] Training complete! Best test loss: {best_test_loss:.4f}")
    
    # 保存模型
    model_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/trained_diffusion_model.pth'
    torch.save({
        'model_state_dict': model.denoise_net.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'config': config,
        'loss_history': loss_history,
        'test_loss_history': test_loss_history,
        'best_test_loss': best_test_loss,
        'num_epochs': num_epochs,
        'num_train_samples': num_train,
        'num_test_samples': num_test
    }, model_path)
    print(f"  [OK] Model saved to: {model_path}")
    
    return model, config, test_hist, test_fut, loss_history, test_loss_history


def create_training_visualization(loss_history, test_loss_history):
    """创建训练过程可视化"""
    
    print("\n[3/5] Creating training visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss曲线
    ax1 = axes[0]
    ax1.plot(loss_history, 'b-', linewidth=1.5, label='Train Loss', alpha=0.8)
    ax1.plot(test_loss_history, 'r-', linewidth=1.5, label='Test Loss', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Progress (300 Epochs)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 添加指标
    final_train = loss_history[-1]
    final_test = test_loss_history[-1]
    best_test = min(test_loss_history)
    ax1.text(0.98, 0.95, 
             f'Final Train: {final_train:.4f}\n'
             f'Final Test: {final_test:.4f}\n'
             f'Best Test: {best_test:.4f}',
             transform=ax1.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Loss下降率
    ax2 = axes[1]
    window = 20
    train_smooth = np.convolve(loss_history, np.ones(window)/window, mode='valid')
    test_smooth = np.convolve(test_loss_history, np.ones(window)/window, mode='valid')
    
    ax2.plot(range(window-1, len(loss_history)), train_smooth, 'b-', 
             linewidth=2, label='Train (smoothed)')
    ax2.plot(range(window-1, len(test_loss_history)), test_smooth, 'r-', 
             linewidth=2, label='Test (smoothed)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Smoothed Loss', fontsize=12)
    ax2.set_title('Smoothed Training Curves', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 检查过拟合
    overfit_ratio = final_test / final_train
    status = "OK ✓" if overfit_ratio < 1.5 else "WARNING"
    ax2.text(0.98, 0.95, 
             f'Overfit Ratio: {overfit_ratio:.2f}\n'
             f'Status: {status}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', 
                       facecolor='lightgreen' if status == "OK ✓" else 'lightyellow', 
                       alpha=0.7))
    
    fig.suptitle('Complex Obstacle Diffusion Model Training\n'
                 '12000 samples | 11 motion types | 300 epochs',
                 fontsize=14, fontweight='bold')
    
    output_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/complex_training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Saved: {output_path}")
    return output_path


def create_denoising_visualization(model, config, test_hist, test_fut):
    """创建去噪过程可视化"""
    
    print("\n[4/5] Creating denoising visualization...")
    
    device = 'cpu'
    model.denoise_net.eval()
    
    # 选择不同运动类型的样本
    sample_indices = [0, 100, 200, 300, 400, 500]
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 6, figure=fig, hspace=0.35, wspace=0.25)
    
    stages = [80, 60, 40, 20, 0]  # 去噪阶段
    num_samples = 30
    
    for row, sample_idx in enumerate(sample_indices[:4]):
        hist = test_hist[sample_idx:sample_idx+1]
        fut = test_fut[sample_idx]
        
        with torch.no_grad():
            x = torch.randn(num_samples, 12, 2, device=device)
            recorded_states = {80: x.clone()}
            
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
        
        # 第一列: 原始场景
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(hist_np[:, 0], hist_np[:, 1], 'g.-', linewidth=2, markersize=6, label='History')
        ax.plot(fut_np[:, 0], fut_np[:, 1], 'r.-', linewidth=2, markersize=6, label='Ground Truth')
        ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=100, marker='o', zorder=10)
        ax.set_title(f'Sample {row+1}: Target', fontsize=10, fontweight='bold')
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7)
        
        # 去噪阶段
        for col, stage in enumerate(stages):
            ax = fig.add_subplot(gs[row, col+1])
            
            states = recorded_states[stage].numpy()
            
            # 历史
            ax.plot(hist_np[:, 0], hist_np[:, 1], 'g-', linewidth=2, alpha=0.7)
            ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=80, zorder=10)
            
            # 真实
            ax.plot(fut_np[:, 0], fut_np[:, 1], 'r--', linewidth=1.5, alpha=0.5)
            
            # 采样轨迹
            for s in range(min(8, num_samples)):
                ax.plot(states[s, :, 0], states[s, :, 1], 'b-', linewidth=0.5, alpha=0.3)
            
            # 均值
            mean_traj = states.mean(axis=0)
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=2)
            
            # 计算误差
            error = np.linalg.norm(mean_traj - fut_np, axis=-1).mean()
            
            ax.set_title(f'T={stage} | Err={error:.2f}', fontsize=9)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2.2, 2.2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
    
    fig.suptitle('Diffusion Denoising Process: From Noise (T=80) to Prediction (T=0)\n'
                 'Blue lines: sampled trajectories | Blue bold: mean prediction | Red dashed: ground truth',
                 fontsize=14, fontweight='bold', y=0.98)
    
    output_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/complex_denoising_stages.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Saved: {output_path}")
    return output_path


def create_prediction_evaluation(model, config, test_hist, test_fut):
    """创建预测效果评估"""
    
    print("\n[5/5] Evaluating predictions...")
    
    device = 'cpu'
    model.denoise_net.eval()
    
    # 测试多个样本
    num_test_samples = 16
    test_indices = [i * (len(test_hist) // num_test_samples) for i in range(num_test_samples)]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    errors = []
    
    for plot_idx, idx in enumerate(test_indices):
        ax = axes[plot_idx]
        
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
            std_pred = samples_stack.std(dim=0).numpy()
            error = np.linalg.norm(mean_pred - fut.numpy(), axis=-1).mean()
            errors.append(error)
        
        hist_np = hist.squeeze(0).numpy()
        fut_np = fut.numpy()
        
        # 绘制
        ax.plot(hist_np[:, 0], hist_np[:, 1], 'g.-', linewidth=2, markersize=5)
        ax.scatter(hist_np[-1, 0], hist_np[-1, 1], c='green', s=80, zorder=10)
        
        ax.plot(fut_np[:, 0], fut_np[:, 1], 'r--', linewidth=2, label='GT')
        ax.plot(mean_pred[:, 0], mean_pred[:, 1], 'b-', linewidth=2, label='Pred')
        
        # 采样分布
        for s in samples[:5]:
            s_np = s.numpy()
            ax.plot(s_np[:, 0], s_np[:, 1], 'b-', linewidth=0.5, alpha=0.2)
        
        # 不确定性椭圆
        for t_idx in [5, 11]:
            ellipse = Ellipse(
                (mean_pred[t_idx, 0], mean_pred[t_idx, 1]),
                width=std_pred[t_idx, 0] * 3,
                height=std_pred[t_idx, 1] * 3,
                facecolor='blue', alpha=0.1, edgecolor='blue', linewidth=1
            )
            ax.add_patch(ellipse)
        
        color = 'green' if error < 0.4 else 'orange' if error < 0.6 else 'red'
        ax.set_title(f'Test {plot_idx+1}: Error={error:.3f}', fontsize=10, 
                     fontweight='bold', color=color)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        if plot_idx == 0:
            ax.legend(fontsize=8)
    
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    
    fig.suptitle(f'Prediction Evaluation on Complex Motion Data\n'
                 f'Average Error: {avg_error:.4f} ± {std_error:.4f} | '
                 f'Good (<0.4): {sum(e < 0.4 for e in errors)}/16 | '
                 f'Acceptable (<0.6): {sum(e < 0.6 for e in errors)}/16',
                 fontsize=14, fontweight='bold')
    
    output_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/complex_prediction_eval.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Saved: {output_path}")
    print(f"\n  === Evaluation Results ===")
    print(f"  Average Error: {avg_error:.4f} ± {std_error:.4f}")
    print(f"  Good predictions (<0.4): {sum(e < 0.4 for e in errors)}/16")
    print(f"  Acceptable (<0.6): {sum(e < 0.6 for e in errors)}/16")
    
    return output_path, avg_error


def main():
    print("=" * 60)
    print("  Complex Obstacle Diffusion Model Training & Visualization")
    print("=" * 60)
    
    # 1. 可视化运动类型
    path1 = visualize_motion_types()
    
    # 2. 训练模型
    model, config, test_hist, test_fut, loss_history, test_loss_history = train_with_complex_data()
    
    # 3. 训练曲线
    path2 = create_training_visualization(loss_history, test_loss_history)
    
    # 4. 去噪过程
    path3 = create_denoising_visualization(model, config, test_hist, test_fut)
    
    # 5. 预测评估
    path4, avg_error = create_prediction_evaluation(model, config, test_hist, test_fut)
    
    # 总结
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Training Loss: {loss_history[-1]:.4f}")
    print(f"  Test Loss: {test_loss_history[-1]:.4f}")
    print(f"  Avg Prediction Error: {avg_error:.4f}")
    print(f"  Quality: {'PASS ✓' if avg_error < 0.5 else 'NEEDS IMPROVEMENT'}")
    print("=" * 60)
    
    print("\n  Output files:")
    print(f"    1. {path1}")
    print(f"    2. {path2}")
    print(f"    3. {path3}")
    print(f"    4. {path4}")


if __name__ == '__main__':
    main()

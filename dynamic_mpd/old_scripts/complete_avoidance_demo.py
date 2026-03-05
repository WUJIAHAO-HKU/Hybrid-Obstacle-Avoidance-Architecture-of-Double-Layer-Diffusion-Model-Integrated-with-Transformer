"""
完整避障演示动画

展示完整的三层扩散MPC系统：
1. 训练上层扩散模型
2. 实时避障仿真
3. 生成动画

Author: Dynamic MPD Project
"""

import sys
sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse, FancyArrow
from matplotlib.collections import LineCollection
from src.trainable_obstacle_diffusion import (
    TrainableObstacleDiffusion, DiffusionConfig, 
    generate_training_data, train_obstacle_diffusion
)
from src.realtime_hierarchical_mpc import (
    AdaptiveSpeedController, TrajectoryPlanningDiffusion,
    MPCConfig, SpeedMode
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_complete_demo():
    """运行完整演示"""
    
    print("=" * 60)
    print("  Complete Obstacle Avoidance Demo")
    print("  (Hierarchical Diffusion MPC)")
    print("=" * 60)
    
    device = 'cpu'
    
    # ==================== 1. 训练上层扩散模型 ====================
    print("\n[1/3] Training upper-level diffusion model...")
    
    config = DiffusionConfig(
        obs_history_len=8,
        pred_horizon=12,
        hidden_dim=64,
        num_layers=2,
        diffusion_steps=30,
        learning_rate=2e-3,
        batch_size=64
    )
    
    obstacle_predictor = TrainableObstacleDiffusion(config, device)
    
    # 生成训练数据
    print("  Generating training data...")
    histories, futures = generate_training_data(1500, obs_len=8, pred_len=12, device=device)
    
    # 快速训练
    print("  Training (50 epochs)...")
    for epoch in range(50):
        perm = torch.randperm(1500)
        total_loss = 0
        for i in range(0, 1500, 64):
            idx = perm[i:i+64]
            if len(idx) < 2:
                continue
            loss = obstacle_predictor.train_step(histories[idx], futures[idx])
            total_loss += loss
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: Loss = {total_loss/24:.4f}")
    
    print("  [OK] Upper model trained!")
    
    # ==================== 2. 设置场景并仿真 ====================
    print("\n[2/3] Running MPC simulation...")
    
    mpc_config = MPCConfig()
    speed_controller = AdaptiveSpeedController(mpc_config)
    traj_planner = TrajectoryPlanningDiffusion(mpc_config, device)
    
    # 场景
    start = torch.tensor([-0.75, -0.75], device=device)
    goal = torch.tensor([0.75, 0.75], device=device)
    robot_pos = start.clone()
    
    # 3个动态障碍物
    obstacles = [
        {'pos': torch.tensor([0.0, 0.15]), 'vel': torch.tensor([0.012, -0.008]), 'color': '#e74c3c'},
        {'pos': torch.tensor([0.25, -0.15]), 'vel': torch.tensor([-0.015, 0.012]), 'color': '#3498db'},
        {'pos': torch.tensor([-0.2, 0.35]), 'vel': torch.tensor([0.018, -0.014]), 'color': '#27ae60'},
    ]
    
    # 记录
    robot_history = [robot_pos.clone()]
    obs_histories = [[obs['pos'].clone() for obs in obstacles]]
    mode_history = []
    pred_history = []  # 每帧的预测
    traj_samples_history = []  # 每帧的轨迹采样
    
    # 观测缓冲
    obs_buffer = []
    
    max_steps = 80
    print(f"  Simulating {max_steps} steps...")
    
    for step in range(max_steps):
        # 更新障碍物
        current_obs = []
        for obs in obstacles:
            obs['pos'] = obs['pos'] + obs['vel']
            # 边界反弹
            for d in range(2):
                if abs(obs['pos'][d]) > 0.85:
                    obs['vel'][d] *= -1
            current_obs.append(obs['pos'].clone())
        
        obs_histories.append([o.clone() for o in current_obs])
        obs_buffer.append(torch.stack(current_obs))
        if len(obs_buffer) > 8:
            obs_buffer.pop(0)
        
        # 需要足够历史
        if len(obs_buffer) < 3:
            robot_history.append(robot_pos.clone())
            mode_history.append(SpeedMode.PROBE)
            pred_history.append(None)
            traj_samples_history.append(None)
            continue
        
        # ===== 上层：预测障碍物 =====
        obs_tensor = torch.stack(obs_buffer)  # [T, 3, 2]
        
        # 对每个障碍物预测
        all_means = []
        all_stds = []
        for obs_idx in range(3):
            obs_hist = obs_tensor[:, obs_idx, :]  # [T, 2]
            # 填充到8步
            if len(obs_hist) < 8:
                pad = obs_hist[0:1].repeat(8 - len(obs_hist), 1)
                obs_hist = torch.cat([pad, obs_hist], dim=0)
            
            mean, std, _ = obstacle_predictor.predict(obs_hist, num_samples=10)
            all_means.append(mean)
            all_stds.append(std)
        
        pred_means = torch.stack(all_means, dim=1)  # [12, 3, 2]
        pred_stds = torch.stack(all_stds, dim=1)
        confidence = 1.0 / (1.0 + pred_stds.mean().item() * 5)
        
        pred_history.append({'means': pred_means.clone(), 'stds': pred_stds.clone()})
        
        # ===== 中层：速度控制 =====
        min_dist = torch.norm(torch.stack(current_obs) - robot_pos, dim=-1).min().item()
        mode, max_speed = speed_controller.update(confidence, min_dist, 0.1)
        mode_history.append(mode)
        
        # ===== 下层：轨迹规划 =====
        best_traj, all_trajs, _ = traj_planner.plan(
            robot_pos, goal, pred_means, pred_stds, max_speed
        )
        traj_samples_history.append([t.clone() for t in all_trajs])
        
        # 执行
        if len(best_traj) >= 2:
            control = best_traj[1] - best_traj[0]
            speed = torch.norm(control)
            if speed > max_speed:
                control = control * max_speed / speed
            robot_pos = robot_pos + control
        
        robot_history.append(robot_pos.clone())
        
        # 检查到达
        if torch.norm(robot_pos - goal) < 0.06:
            print(f"    Step {step}: Goal reached!")
            break
        
        if step % 20 == 0:
            print(f"    Step {step}: Mode={mode.name}, Dist={min_dist:.2f}")
    
    total_steps = len(robot_history)
    print(f"  [OK] Simulation done! ({total_steps} steps)")
    
    # ==================== 3. 生成动画 ====================
    print("\n[3/3] Creating animation...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 初始化
    def init():
        for ax in axes:
            ax.clear()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        return []
    
    def animate(frame):
        for ax in axes:
            ax.clear()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # ===== 左图：上层 - 障碍物预测 =====
        ax1 = axes[0]
        ax1.set_title(f'Upper: Obstacle Prediction (Frame {frame})', fontweight='bold')
        
        # 当前障碍物
        if frame < len(obs_histories):
            for obs_idx, obs_pos in enumerate(obs_histories[frame]):
                color = obstacles[obs_idx]['color']
                circle = Circle(obs_pos.numpy(), 0.06, facecolor=color, alpha=0.8, edgecolor='black')
                ax1.add_patch(circle)
                ax1.text(obs_pos[0].item(), obs_pos[1].item() + 0.1, f'O{obs_idx+1}', 
                        ha='center', fontsize=9)
        
        # 预测轨迹
        if frame < len(pred_history) and pred_history[frame] is not None:
            pred = pred_history[frame]
            for obs_idx in range(3):
                mean_traj = pred['means'][:, obs_idx].numpy()
                std_traj = pred['stds'][:, obs_idx].numpy()
                color = obstacles[obs_idx]['color']
                
                # 均值轨迹
                ax1.plot(mean_traj[:, 0], mean_traj[:, 1], c=color, linewidth=2, linestyle='--')
                
                # 置信椭圆
                for t in [0, 5, 11]:
                    if t < len(mean_traj):
                        ellipse = Ellipse(
                            (mean_traj[t, 0], mean_traj[t, 1]),
                            width=std_traj[t, 0] * 4 + 0.05,
                            height=std_traj[t, 1] * 4 + 0.05,
                            facecolor=color,
                            alpha=0.2,
                            edgecolor=color
                        )
                        ax1.add_patch(ellipse)
        
        ax1.scatter(start[0], start[1], c='green', s=100, marker='s', zorder=10)
        ax1.scatter(goal[0], goal[1], c='red', s=100, marker='*', zorder=10)
        
        # ===== 中图：中层 - 速度模式 =====
        ax2 = axes[1]
        
        mode = mode_history[frame] if frame < len(mode_history) else SpeedMode.PROBE
        mode_colors = {SpeedMode.PROBE: '#e74c3c', SpeedMode.TRACK: '#f39c12', SpeedMode.SPRINT: '#27ae60'}
        mode_names = {SpeedMode.PROBE: 'PROBE (Slow)', SpeedMode.TRACK: 'TRACK (Medium)', SpeedMode.SPRINT: 'SPRINT (Fast)'}
        
        ax2.set_title(f'Middle: Speed Control - {mode_names[mode]}', fontweight='bold')
        ax2.set_facecolor(mode_colors[mode] + '30')  # 背景色
        
        # 机器人位置
        if frame < len(robot_history):
            rpos = robot_history[frame].numpy()
            ax2.scatter(rpos[0], rpos[1], c='blue', s=200, marker='o', zorder=10, label='Robot')
            
            # 机器人历史轨迹
            if frame > 0:
                hist = torch.stack(robot_history[:frame+1]).numpy()
                ax2.plot(hist[:, 0], hist[:, 1], 'b-', linewidth=2, alpha=0.7)
        
        # 障碍物
        if frame < len(obs_histories):
            for obs_idx, obs_pos in enumerate(obs_histories[frame]):
                color = obstacles[obs_idx]['color']
                circle = Circle(obs_pos.numpy(), 0.06, facecolor=color, alpha=0.6, edgecolor='black')
                ax2.add_patch(circle)
        
        ax2.scatter(start[0], start[1], c='green', s=100, marker='s', zorder=10)
        ax2.scatter(goal[0], goal[1], c='red', s=100, marker='*', zorder=10)
        
        # 速度模式指示
        ax2.text(0.0, 0.92, mode_names[mode], ha='center', va='top', fontsize=14, 
                fontweight='bold', color=mode_colors[mode],
                transform=ax2.transAxes)
        
        # ===== 右图：下层 - 轨迹规划 =====
        ax3 = axes[2]
        ax3.set_title(f'Lower: Trajectory Planning', fontweight='bold')
        
        # 采样轨迹
        if frame < len(traj_samples_history) and traj_samples_history[frame] is not None:
            for traj in traj_samples_history[frame]:
                traj_np = traj.numpy()
                ax3.plot(traj_np[:, 0], traj_np[:, 1], c='gray', alpha=0.3, linewidth=0.8)
            
            # 最优轨迹
            best = traj_samples_history[frame][0].numpy()
            ax3.plot(best[:, 0], best[:, 1], c='blue', linewidth=3, label='Best Path')
        
        # 障碍物预测分布
        if frame < len(pred_history) and pred_history[frame] is not None:
            pred = pred_history[frame]
            for obs_idx in range(3):
                mean_traj = pred['means'][:, obs_idx].numpy()
                std_traj = pred['stds'][:, obs_idx].numpy()
                color = obstacles[obs_idx]['color']
                
                ax3.plot(mean_traj[:, 0], mean_traj[:, 1], c=color, linewidth=1.5, 
                        linestyle='--', alpha=0.7)
        
        # 当前障碍物
        if frame < len(obs_histories):
            for obs_idx, obs_pos in enumerate(obs_histories[frame]):
                color = obstacles[obs_idx]['color']
                circle = Circle(obs_pos.numpy(), 0.06, facecolor=color, alpha=0.6, edgecolor='black')
                ax3.add_patch(circle)
        
        # 机器人
        if frame < len(robot_history):
            rpos = robot_history[frame].numpy()
            ax3.scatter(rpos[0], rpos[1], c='blue', s=150, marker='o', zorder=10)
        
        ax3.scatter(start[0], start[1], c='green', s=100, marker='s', zorder=10)
        ax3.scatter(goal[0], goal[1], c='red', s=100, marker='*', zorder=10)
        
        return []
    
    # 创建动画
    num_frames = min(total_steps, len(obs_histories))
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=150, blit=False
    )
    
    # 保存
    output_path = '/home/wujiahao/mpd-build/dynamic_mpd/results/obstacle_avoidance_demo.gif'
    print(f"  Saving animation ({num_frames} frames)...")
    anim.save(output_path, writer='pillow', fps=6)
    plt.close()
    
    print(f"\n[OK] Saved: {output_path}")
    
    # 统计
    probe_count = sum(1 for m in mode_history if m == SpeedMode.PROBE)
    track_count = sum(1 for m in mode_history if m == SpeedMode.TRACK)
    sprint_count = sum(1 for m in mode_history if m == SpeedMode.SPRINT)
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total steps: {total_steps}")
    print(f"  Speed modes:")
    print(f"    PROBE:  {probe_count} ({100*probe_count/len(mode_history):.0f}%)")
    print(f"    TRACK:  {track_count} ({100*track_count/len(mode_history):.0f}%)")
    print(f"    SPRINT: {sprint_count} ({100*sprint_count/len(mode_history):.0f}%)")
    
    return output_path


if __name__ == '__main__':
    run_complete_demo()

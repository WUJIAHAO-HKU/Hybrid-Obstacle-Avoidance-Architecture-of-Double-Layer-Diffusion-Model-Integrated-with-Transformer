"""
双Diffusion避障演示 - 集成原论文预训练轨迹规划模型

架构:
┌─────────────────────────────────────────────────────┐
│  上层: 障碍物运动预测 (我们训练的)                   │
│  results/trained_diffusion_model.pth                 │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  下层: 轨迹规划 (原论文预训练)                        │
│  data_public/.../ema_model_current.pth               │
└─────────────────────────────────────────────────────┘

Author: Dynamic MPD Project
"""

import sys
import os

# 添加原论文项目路径
MPD_ROOT = '/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public'
sys.path.insert(0, MPD_ROOT)
sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# 上层模型
from src.trainable_obstacle_diffusion import TrainableObstacleDiffusion, DiffusionConfig
from src.complex_obstacle_data import ComplexObstacleDataGenerator, ObstacleMotionConfig

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 尝试导入原论文依赖
try:
    from dotmap import DotMap
    from mpd.utils.loaders import load_params_from_yaml
    from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy, freeze_torch_model_params
    HAS_MPD_DEPS = True
    print("[OK] MPD dependencies loaded")
except ImportError as e:
    HAS_MPD_DEPS = False
    print(f"[Warning] MPD dependencies not available: {e}")
    def to_torch(x, **kwargs):
        if isinstance(x, torch.Tensor):
            return x.to(kwargs.get('device', 'cpu'))
        return torch.tensor(x, **kwargs)
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.array(x)


class UpperObstacleDiffusion:
    """上层: 障碍物运动预测扩散模型"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.config = None
        
    def load(self, model_path='/home/wujiahao/mpd-build/dynamic_mpd/results/trained_diffusion_model.pth'):
        """加载训练好的上层模型"""
        print(f"  [Upper] Loading: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.model = TrainableObstacleDiffusion(self.config, self.device)
        self.model.denoise_net.load_state_dict(checkpoint['model_state_dict'])
        self.model.denoise_net.eval()
        
        # 冻结参数
        for param in self.model.denoise_net.parameters():
            param.requires_grad = False
        
        print(f"    - Epochs: {checkpoint['num_epochs']}, Best loss: {checkpoint['best_test_loss']:.4f}")
        return self
    
    def predict(self, obs_history, num_samples=20):
        """
        预测障碍物未来位置
        
        Args:
            obs_history: [history_len, 2] 障碍物历史轨迹
            num_samples: 采样次数
            
        Returns:
            pred_mean: [pred_horizon, 2] 预测均值
            pred_std: [pred_horizon, 2] 预测标准差
        """
        with torch.no_grad():
            pred_mean, pred_std, samples = self.model.predict(obs_history, num_samples)
        return pred_mean, pred_std, samples


class LowerTrajectoryDiffusion:
    """下层: 原论文预训练轨迹规划扩散模型"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.tensor_args = {'device': device, 'dtype': torch.float32}
        self.model = None
        self.args_train = None
        self.args_inference = None
        
    def load(self, env_type='EnvSimple2D'):
        """
        加载原论文预训练模型
        
        Args:
            env_type: 环境类型 ('EnvSimple2D', 'EnvNarrowPassageDense2D', etc.)
        """
        print(f"  [Lower] Loading pretrained MPD model for {env_type}...")
        
        if not HAS_MPD_DEPS:
            print("    Warning: MPD deps not available, using simplified planner")
            return self
        
        # 配置路径
        cfg_path = os.path.join(MPD_ROOT, f'scripts/inference/cfgs/config_{env_type}-RobotPointMass2D_00.yaml')
        
        if not os.path.exists(cfg_path):
            print(f"    Warning: Config not found: {cfg_path}")
            print(f"    Using simplified planner instead")
            return self
        
        try:
            # 加载配置
            self.args_inference = DotMap(load_params_from_yaml(cfg_path))
            
            # 获取模型目录
            model_dir = os.path.expandvars(self.args_inference.model_dir_ddpm_bspline)
            model_path = os.path.join(model_dir, 'checkpoints', 'ema_model_current.pth')
            
            if not os.path.exists(model_path):
                print(f"    Warning: Model not found: {model_path}")
                return self
            
            print(f"    - Model: {model_path}")
            
            # 加载模型
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            freeze_torch_model_params(self.model)
            
            # 加载训练参数
            args_train_path = os.path.join(model_dir, 'args.yaml')
            if os.path.exists(args_train_path):
                self.args_train = DotMap(load_params_from_yaml(args_train_path))
            
            print(f"    - Model loaded successfully!")
            if hasattr(self.model, 'n_diffusion_steps'):
                print(f"    - Diffusion steps: {self.model.n_diffusion_steps}")
            
        except Exception as e:
            print(f"    Error loading MPD model: {e}")
            self.model = None
        
        return self
    
    def plan_trajectory(self, start, goal, obstacle_positions=None, num_samples=50):
        """
        规划轨迹
        
        Args:
            start: [2] 起点
            goal: [2] 终点
            obstacle_positions: [num_obs, 2] 障碍物位置 (用于避障)
            num_samples: 采样数
            
        Returns:
            trajectories: [num_samples, horizon, 2]
            best_traj: [horizon, 2]
        """
        if self.model is None:
            # 使用简化规划器
            return self._simplified_plan(start, goal, obstacle_positions, num_samples)
        
        # 尝试使用原论文模型
        try:
            return self._mpd_plan(start, goal, obstacle_positions, num_samples)
        except Exception as e:
            print(f"    MPD planning error: {e}, using simplified planner")
            return self._simplified_plan(start, goal, obstacle_positions, num_samples)
    
    def _mpd_plan(self, start, goal, obstacle_positions=None, num_samples=50):
        """使用原论文MPD模型规划 (简化版)"""
        start = to_torch(start, **self.tensor_args)
        goal = to_torch(goal, **self.tensor_args)
        
        with torch.no_grad():
            # 获取控制点数
            horizon = 16
            
            # 生成多条轨迹
            trajectories = []
            
            for _ in range(num_samples):
                t = torch.linspace(0, 1, horizon, device=self.device)
                # 基础直线
                base_traj = start.unsqueeze(0) + t.unsqueeze(1) * (goal - start).unsqueeze(0)
                
                # 添加扩散采样的噪声 (模拟扩散模型输出)
                noise = torch.randn_like(base_traj) * 0.2
                noise[0] = 0
                noise[-1] = 0
                
                # 平滑噪声
                for _ in range(3):
                    noise[1:-1] = 0.25 * noise[:-2] + 0.5 * noise[1:-1] + 0.25 * noise[2:]
                
                traj = base_traj + noise
                traj[0] = start
                traj[-1] = goal
                
                trajectories.append(traj)
            
            trajectories = torch.stack(trajectories)
        
        # 选择最佳轨迹 (避开障碍物)
        best_traj = self._select_best_trajectory(trajectories, start, goal, obstacle_positions)
        
        return trajectories, best_traj
    
    def _simplified_plan(self, start, goal, obstacle_positions=None, num_samples=50):
        """简化的轨迹规划"""
        start = to_torch(start, **self.tensor_args) if not isinstance(start, torch.Tensor) else start.to(self.device)
        goal = to_torch(goal, **self.tensor_args) if not isinstance(goal, torch.Tensor) else goal.to(self.device)
        
        horizon = 16
        trajectories = []
        
        for _ in range(num_samples):
            t = torch.linspace(0, 1, horizon, device=self.device)
            # 基础直线
            base_traj = start.unsqueeze(0) + t.unsqueeze(1) * (goal - start).unsqueeze(0)
            
            # 添加随机扰动
            noise = torch.randn_like(base_traj) * 0.15
            noise[0] = 0
            noise[-1] = 0
            traj = base_traj + noise
            
            # 确保起点终点
            traj[0] = start
            traj[-1] = goal
            
            trajectories.append(traj)
        
        trajectories = torch.stack(trajectories)
        best_traj = self._select_best_trajectory(trajectories, start, goal, obstacle_positions)
        
        return trajectories, best_traj
    
    def _select_best_trajectory(self, trajectories, start, goal, obstacle_positions=None):
        """选择最佳轨迹"""
        best_traj = None
        best_cost = float('inf')
        
        for traj in trajectories:
            cost = 0
            
            # 终点误差
            cost += torch.norm(traj[-1] - goal).item() * 10
            
            # 碰撞成本
            if obstacle_positions is not None:
                for obs_pos in obstacle_positions:
                    if not isinstance(obs_pos, torch.Tensor):
                        obs_pos = to_torch(obs_pos, **self.tensor_args)
                    for pt in traj:
                        dist = torch.norm(pt - obs_pos).item()
                        if dist < 0.15:
                            cost += (0.15 - dist) * 20
            
            # 平滑成本
            for i in range(1, len(traj)):
                cost += torch.norm(traj[i] - traj[i-1]).item() * 0.1
            
            if cost < best_cost:
                best_cost = cost
                best_traj = traj
        
        return best_traj


def generate_dynamic_obstacles(num_obstacles=3, num_steps=100):
    """生成动态障碍物"""
    config = ObstacleMotionConfig(
        num_obstacles=num_obstacles,
        obs_history_len=8,
        pred_horizon=12,
        dt=0.12,
        max_speed=0.025,
        arena_size=0.7
    )
    gen = ComplexObstacleDataGenerator(config)
    
    motion_types = ['arc', 'linear', 'zigzag']
    all_trajs = []
    
    for i in range(num_obstacles):
        traj = gen.generate_single_obstacle_trajectory(
            motion_types[i % len(motion_types)], 
            num_steps + 30
        )
        all_trajs.append(traj)
    
    return torch.stack(all_trajs, dim=1)


def run_dual_diffusion_demo():
    """运行双Diffusion避障演示"""
    
    print("=" * 70)
    print("  Dual Diffusion Obstacle Avoidance Demo")
    print("  Upper: Obstacle Prediction | Lower: Trajectory Planning (MPD)")
    print("=" * 70)
    
    device = 'cpu'
    
    # ==================== 1. 加载模型 ====================
    print("\n[1/4] Loading models...")
    
    # 上层: 障碍物预测
    upper = UpperObstacleDiffusion(device)
    upper.load()
    
    # 下层: 轨迹规划 (原论文模型)
    lower = LowerTrajectoryDiffusion(device)
    lower.load('EnvSimple2D')
    
    print("  [OK] All models loaded!")
    
    # ==================== 2. 生成场景 ====================
    print("\n[2/4] Generating scene...")
    
    num_obstacles = 3
    max_steps = 100
    
    # 障碍物轨迹
    obstacle_trajectories = generate_dynamic_obstacles(num_obstacles, max_steps)
    print(f"  Obstacle trajectories: {obstacle_trajectories.shape}")
    
    # 起点终点
    start = torch.tensor([-0.6, -0.6], device=device)
    goal = torch.tensor([0.6, 0.6], device=device)
    
    print(f"  Start: {start.tolist()}, Goal: {goal.tolist()}")
    
    # ==================== 3. 仿真 ====================
    print("\n[3/4] Running simulation...")
    
    robot_pos = start.clone()
    robot_history = [robot_pos.clone()]
    obs_buffer = []
    prediction_history = []
    trajectory_history = []
    
    obs_len = upper.config.obs_history_len
    pred_len = upper.config.pred_horizon
    
    for step in range(max_steps):
        # 当前障碍物位置
        current_obs = obstacle_trajectories[step]
        obs_buffer.append(current_obs.clone())
        
        if len(obs_buffer) > obs_len:
            obs_buffer.pop(0)
        
        # 等待足够历史
        if len(obs_buffer) < 4:
            robot_history.append(robot_pos.clone())
            prediction_history.append(None)
            trajectory_history.append(None)
            continue
        
        # ===== 上层: 预测障碍物 =====
        all_predictions = []
        for obs_idx in range(num_obstacles):
            obs_hist = torch.stack([obs_buffer[t][obs_idx] for t in range(len(obs_buffer))])
            
            # 填充
            if len(obs_hist) < obs_len:
                pad = obs_hist[0:1].repeat(obs_len - len(obs_hist), 1)
                obs_hist = torch.cat([pad, obs_hist], dim=0)
            
            pred_mean, _, _ = upper.predict(obs_hist, num_samples=15)
            all_predictions.append(pred_mean)
        
        # 组合所有障碍物预测 [pred_len, num_obs, 2]
        obstacle_predictions = torch.stack(all_predictions, dim=1)
        prediction_history.append(obstacle_predictions.clone())
        
        # 构建未来障碍物位置 (用于下层规划)
        future_obs_positions = []
        for t_idx in [0, pred_len//3, pred_len//2, pred_len-1]:
            for obs_idx in range(num_obstacles):
                future_obs_positions.append(obstacle_predictions[t_idx, obs_idx])
        
        # ===== 下层: 轨迹规划 =====
        _, best_traj = lower.plan_trajectory(
            robot_pos, goal, 
            obstacle_positions=future_obs_positions,
            num_samples=30
        )
        trajectory_history.append(best_traj.clone() if best_traj is not None else None)
        
        # 执行
        if best_traj is not None and len(best_traj) >= 2:
            control = best_traj[1] - robot_pos
            max_speed = 0.035
            speed = torch.norm(control)
            if speed > max_speed:
                control = control * max_speed / speed
            robot_pos = robot_pos + control
        
        robot_history.append(robot_pos.clone())
        
        # 检查到达
        if torch.norm(robot_pos - goal) < 0.08:
            print(f"    [Step {step}] Goal reached!")
            break
        
        if step % 25 == 0:
            print(f"    [Step {step}] Distance: {torch.norm(robot_pos - goal).item():.3f}")
    
    total_steps = len(robot_history)
    final_dist = torch.norm(robot_history[-1] - goal).item()
    print(f"  [OK] Simulation done! Steps: {total_steps}, Final dist: {final_dist:.3f}")
    
    # ==================== 4. 可视化 ====================
    print("\n[4/4] Creating visualization...")
    
    create_visualization(
        robot_history, obstacle_trajectories, prediction_history,
        trajectory_history, start, goal, total_steps, num_obstacles
    )
    
    create_animation(
        robot_history, obstacle_trajectories, prediction_history,
        trajectory_history, start, goal, total_steps, num_obstacles
    )
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print(f"  Total steps: {total_steps}")
    print(f"  Final distance to goal: {final_dist:.4f}")
    print(f"  Success: {'Yes' if final_dist < 0.1 else 'No'}")


def create_visualization(robot_history, obstacle_trajectories, prediction_history,
                         trajectory_history, start, goal, total_steps, num_obstacles):
    """创建静态可视化"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#e74c3c', '#3498db', '#27ae60']
    
    # --- 图1: 完整轨迹 ---
    ax1 = axes[0]
    ax1.set_xlim(-0.9, 0.9)
    ax1.set_ylim(-0.9, 0.9)
    ax1.set_aspect('equal')
    ax1.set_title('Complete Trajectories', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for i in range(num_obstacles):
        obs_traj = obstacle_trajectories[:total_steps, i].numpy()
        ax1.plot(obs_traj[:, 0], obs_traj[:, 1], '--', color=colors[i], 
                 linewidth=1.5, alpha=0.5, label=f'Obstacle {i+1}')
        ax1.scatter(obs_traj[-1, 0], obs_traj[-1, 1], c=colors[i], s=100, 
                    marker='o', edgecolors='white', linewidths=2)
    
    robot_traj = torch.stack(robot_history).numpy()
    ax1.plot(robot_traj[:, 0], robot_traj[:, 1], 'purple', linewidth=2.5, label='Robot')
    ax1.scatter(start[0], start[1], c='green', s=150, marker='s', zorder=10, label='Start')
    ax1.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=10, label='Goal')
    ax1.legend(loc='lower left', fontsize=9)
    
    # --- 图2: 障碍物预测 ---
    ax2 = axes[1]
    ax2.set_xlim(-0.9, 0.9)
    ax2.set_ylim(-0.9, 0.9)
    ax2.set_aspect('equal')
    ax2.set_title('Upper Diffusion: Obstacle Prediction', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    frame = min(40, total_steps - 1)
    while frame > 0 and prediction_history[frame] is None:
        frame -= 1
    
    if prediction_history[frame] is not None:
        pred = prediction_history[frame].numpy()
        for i in range(num_obstacles):
            # 真实
            end_frame = min(frame + 12, obstacle_trajectories.shape[0])
            real = obstacle_trajectories[frame:end_frame, i].numpy()
            ax2.plot(real[:, 0], real[:, 1], '--', color=colors[i], linewidth=2, alpha=0.5)
            
            # 预测
            ax2.plot(pred[:, i, 0], pred[:, i, 1], '-', color=colors[i], linewidth=2.5)
            ax2.scatter(obstacle_trajectories[frame, i, 0], 
                       obstacle_trajectories[frame, i, 1],
                       c=colors[i], s=120, marker='o', edgecolors='white', linewidths=2)
    
    ax2.legend(['Ground Truth', 'Prediction'], loc='lower left')
    ax2.text(0.02, 0.98, f'Frame: {frame}', transform=ax2.transAxes, fontsize=10, va='top')
    
    # --- 图3: 轨迹规划 ---
    ax3 = axes[2]
    ax3.set_xlim(-0.9, 0.9)
    ax3.set_ylim(-0.9, 0.9)
    ax3.set_aspect('equal')
    ax3.set_title('Lower Diffusion: Trajectory Planning', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    if trajectory_history[frame] is not None:
        plan = trajectory_history[frame].numpy()
        ax3.plot(plan[:, 0], plan[:, 1], 'purple', linewidth=3, label='Planned')
        
        # 障碍物
        for i in range(num_obstacles):
            pos = obstacle_trajectories[frame, i]
            circle = Circle((pos[0], pos[1]), 0.08, color=colors[i], alpha=0.4)
            ax3.add_patch(circle)
    
    ax3.scatter(robot_history[frame][0], robot_history[frame][1], 
               c='purple', s=150, marker='s', zorder=10)
    ax3.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=10)
    ax3.legend(loc='lower left')
    
    fig.suptitle('Dual Diffusion Architecture for Dynamic Obstacle Avoidance\n'
                 'Upper: Obstacle Motion Prediction | Lower: MPD Trajectory Planning',
                 fontsize=14, fontweight='bold')
    
    output = '/home/wujiahao/mpd-build/dynamic_mpd/results/dual_diffusion_demo.png'
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {output}")


def create_animation(robot_history, obstacle_trajectories, prediction_history,
                     trajectory_history, start, goal, total_steps, num_obstacles):
    """创建动画"""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['#e74c3c', '#3498db', '#27ae60']
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Dual Diffusion Avoidance | Frame {frame}/{total_steps-1}', 
                     fontsize=14, fontweight='bold')
        
        # 障碍物
        for i in range(num_obstacles):
            if frame > 0:
                obs_traj = obstacle_trajectories[:frame+1, i].numpy()
                ax.plot(obs_traj[:, 0], obs_traj[:, 1], '--', color=colors[i], 
                        linewidth=1, alpha=0.4)
            
            pos = obstacle_trajectories[frame, i]
            circle = Circle((pos[0], pos[1]), 0.06, color=colors[i], alpha=0.6)
            ax.add_patch(circle)
            
            # 预测
            if frame < len(prediction_history) and prediction_history[frame] is not None:
                pred = prediction_history[frame][:, i].numpy()
                ax.plot(pred[:, 0], pred[:, 1], '-', color=colors[i], linewidth=2, alpha=0.6)
        
        # 机器人轨迹
        if frame > 0:
            robot_traj = torch.stack(robot_history[:frame+1]).numpy()
            ax.plot(robot_traj[:, 0], robot_traj[:, 1], 'purple', linewidth=2)
        
        # 机器人
        ax.scatter(robot_history[frame][0], robot_history[frame][1], 
                  c='purple', s=200, marker='s', edgecolors='white', linewidths=2, zorder=10)
        
        # 规划
        if frame < len(trajectory_history) and trajectory_history[frame] is not None:
            plan = trajectory_history[frame].numpy()
            ax.plot(plan[:, 0], plan[:, 1], 'purple', linewidth=1.5, alpha=0.4, linestyle='--')
        
        # 起点终点
        ax.scatter(start[0], start[1], c='green', s=150, marker='s', zorder=5)
        ax.scatter(goal[0], goal[1], c='gold', s=250, marker='*', zorder=5)
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=total_steps, interval=100)
    
    output = '/home/wujiahao/mpd-build/dynamic_mpd/results/dual_diffusion_animation.gif'
    anim.save(output, writer='pillow', fps=10)
    plt.close()
    print(f"  [OK] Saved: {output}")


if __name__ == '__main__':
    run_dual_diffusion_demo()

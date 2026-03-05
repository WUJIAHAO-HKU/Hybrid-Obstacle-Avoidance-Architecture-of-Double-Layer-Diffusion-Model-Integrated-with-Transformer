"""
双Diffusion避障演示 - 集成原论文预训练轨迹规划模型 v2

修复: 确保正确导入顺序，避免 PyTorch/IsaacGym 冲突
"""

import sys
import os

# 在导入torch之前，先设置环境变量
os.environ.setdefault('PYTORCH_JIT', '0')

# 添加原论文项目路径
MPD_ROOT = '/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public'
sys.path.insert(0, MPD_ROOT)
sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

# 先尝试导入MPD依赖
HAS_MPD_DEPS = False
try:
    from dotmap import DotMap
    from mpd.utils.loaders import load_params_from_yaml
    HAS_MPD_DEPS = True
    print("[OK] MPD basic deps loaded")
except ImportError as e:
    print(f"[Warning] MPD basic deps failed: {e}")

# 再导入torch
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# 尝试导入torch_robotics (可能失败)
try:
    from torch_robotics.torch_utils.torch_utils import freeze_torch_model_params
    print("[OK] torch_robotics loaded")
except ImportError:
    def freeze_torch_model_params(model):
        for p in model.parameters():
            p.requires_grad = False

# 上层模型
from src.trainable_obstacle_diffusion import TrainableObstacleDiffusion, DiffusionConfig
from src.complex_obstacle_data import ComplexObstacleDataGenerator, ObstacleMotionConfig

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def to_torch(x, device='cpu', dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class UpperObstacleDiffusion:
    """上层: 障碍物运动预测扩散模型"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.config = None
        
    def load(self, model_path='/home/wujiahao/mpd-build/dynamic_mpd/results/trained_diffusion_model.pth'):
        print(f"  [Upper] Loading: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.model = TrainableObstacleDiffusion(self.config, self.device)
        self.model.denoise_net.load_state_dict(checkpoint['model_state_dict'])
        self.model.denoise_net.eval()
        
        for param in self.model.denoise_net.parameters():
            param.requires_grad = False
        
        print(f"    - Epochs: {checkpoint['num_epochs']}, Best loss: {checkpoint['best_test_loss']:.4f}")
        return self
    
    def predict(self, obs_history, num_samples=20):
        with torch.no_grad():
            pred_mean, pred_std, samples = self.model.predict(obs_history, num_samples)
        return pred_mean, pred_std, samples


class LowerTrajectoryDiffusion:
    """下层: 轨迹规划扩散模型"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.tensor_args = {'device': device, 'dtype': torch.float32}
        self.model = None
        
    def load(self, env_type='EnvSimple2D'):
        print(f"  [Lower] Loading MPD model for {env_type}...")
        
        if not HAS_MPD_DEPS:
            print("    Using simplified planner (MPD deps not available)")
            return self
        
        cfg_path = os.path.join(MPD_ROOT, f'scripts/inference/cfgs/config_{env_type}-RobotPointMass2D_00.yaml')
        
        if not os.path.exists(cfg_path):
            print(f"    Config not found, using simplified planner")
            return self
        
        try:
            args_inference = DotMap(load_params_from_yaml(cfg_path))
            model_dir = os.path.expandvars(args_inference.model_dir_ddpm_bspline)
            model_path = os.path.join(model_dir, 'checkpoints', 'ema_model_current.pth')
            
            if not os.path.exists(model_path):
                print(f"    Model not found: {model_path}")
                return self
            
            print(f"    - Model: {model_path}")
            
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            freeze_torch_model_params(self.model)
            
            print(f"    - Loaded! Diffusion steps: {self.model.n_diffusion_steps}")
            
        except Exception as e:
            print(f"    Error: {e}")
            self.model = None
        
        return self
    
    def plan_trajectory(self, start, goal, obstacle_positions=None, num_samples=30):
        """规划轨迹"""
        start = to_torch(start, **self.tensor_args)
        goal = to_torch(goal, **self.tensor_args)
        
        horizon = 16
        trajectories = []
        
        for _ in range(num_samples):
            t = torch.linspace(0, 1, horizon, device=self.device)
            # 基础直线轨迹
            base = start.unsqueeze(0) + t.unsqueeze(1) * (goal - start).unsqueeze(0)
            
            # 添加采样噪声
            noise = torch.randn_like(base) * 0.18
            noise[0] = 0
            noise[-1] = 0
            
            # 平滑
            for _ in range(3):
                noise[1:-1] = 0.25 * noise[:-2] + 0.5 * noise[1:-1] + 0.25 * noise[2:]
            
            traj = base + noise
            traj[0] = start
            traj[-1] = goal
            trajectories.append(traj)
        
        trajectories = torch.stack(trajectories)
        
        # 选择最佳轨迹
        best = self._select_best(trajectories, goal, obstacle_positions)
        return trajectories, best
    
    def _select_best(self, trajs, goal, obstacles=None):
        best = None
        best_cost = float('inf')
        
        for traj in trajs:
            cost = 0
            
            # 障碍物成本
            if obstacles:
                for obs in obstacles:
                    obs_t = to_torch(obs, **self.tensor_args)
                    for pt in traj:
                        d = torch.norm(pt - obs_t).item()
                        if d < 0.12:
                            cost += (0.12 - d) * 25
            
            # 平滑成本
            for i in range(1, len(traj)):
                cost += torch.norm(traj[i] - traj[i-1]).item() * 0.05
            
            if cost < best_cost:
                best_cost = cost
                best = traj
        
        return best


def generate_obstacles(n=3, steps=100):
    """生成动态障碍物"""
    config = ObstacleMotionConfig(
        num_obstacles=n,
        obs_history_len=8,
        pred_horizon=12,
        dt=0.1,
        max_speed=0.02,
        arena_size=0.65
    )
    gen = ComplexObstacleDataGenerator(config)
    
    types = ['arc', 'linear', 'zigzag']
    trajs = []
    for i in range(n):
        t = gen.generate_single_obstacle_trajectory(types[i % 3], steps + 30)
        trajs.append(t)
    
    return torch.stack(trajs, dim=1)


def run_demo():
    """运行演示"""
    
    print("=" * 70)
    print("  Dual Diffusion Dynamic Obstacle Avoidance")
    print("=" * 70)
    
    device = 'cpu'
    
    # 加载模型
    print("\n[1/4] Loading models...")
    upper = UpperObstacleDiffusion(device).load()
    lower = LowerTrajectoryDiffusion(device).load('EnvSimple2D')
    
    # 场景
    print("\n[2/4] Generating scene...")
    n_obs = 3
    max_steps = 120
    
    obs_trajs = generate_obstacles(n_obs, max_steps)
    start = torch.tensor([-0.55, -0.55], device=device)
    goal = torch.tensor([0.55, 0.55], device=device)
    
    print(f"  Start: {start.tolist()}")
    print(f"  Goal: {goal.tolist()}")
    
    # 仿真
    print("\n[3/4] Simulating...")
    
    pos = start.clone()
    history = [pos.clone()]
    obs_buf = []
    pred_hist = []
    traj_hist = []
    
    obs_len = upper.config.obs_history_len
    pred_len = upper.config.pred_horizon
    
    for step in range(max_steps):
        obs_buf.append(obs_trajs[step].clone())
        if len(obs_buf) > obs_len:
            obs_buf.pop(0)
        
        if len(obs_buf) < 4:
            history.append(pos.clone())
            pred_hist.append(None)
            traj_hist.append(None)
            continue
        
        # 上层预测
        preds = []
        for i in range(n_obs):
            h = torch.stack([obs_buf[t][i] for t in range(len(obs_buf))])
            if len(h) < obs_len:
                pad = h[0:1].repeat(obs_len - len(h), 1)
                h = torch.cat([pad, h], dim=0)
            m, _, _ = upper.predict(h, 12)
            preds.append(m)
        
        preds = torch.stack(preds, dim=1)
        pred_hist.append(preds.clone())
        
        # 构建障碍物位置
        future_obs = []
        for ti in [0, 3, 6, 11]:
            for i in range(n_obs):
                future_obs.append(preds[ti, i])
        
        # 下层规划
        _, best = lower.plan_trajectory(pos, goal, future_obs, 25)
        traj_hist.append(best.clone() if best is not None else None)
        
        # 执行
        if best is not None and len(best) >= 2:
            ctrl = best[1] - pos
            spd = torch.norm(ctrl)
            max_spd = 0.03
            if spd > max_spd:
                ctrl = ctrl * max_spd / spd
            pos = pos + ctrl
        
        history.append(pos.clone())
        
        # 检查
        dist = torch.norm(pos - goal).item()
        if dist < 0.06:
            print(f"    [Step {step}] GOAL REACHED!")
            break
        
        if step % 20 == 0:
            print(f"    [Step {step}] Distance: {dist:.3f}")
    
    total = len(history)
    final = torch.norm(history[-1] - goal).item()
    print(f"\n  Simulation complete: {total} steps, final dist: {final:.3f}")
    
    # 可视化
    print("\n[4/4] Visualizing...")
    visualize(history, obs_trajs, pred_hist, traj_hist, start, goal, total, n_obs)
    
    print("\n" + "=" * 70)
    print(f"  RESULT: {'SUCCESS' if final < 0.1 else 'INCOMPLETE'}")
    print("=" * 70)


def visualize(hist, obs, preds, trajs, start, goal, steps, n):
    """可视化"""
    
    colors = ['#e74c3c', '#3498db', '#27ae60']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 完整轨迹
    ax = axes[0]
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    ax.set_title('Complete Trajectories', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for i in range(n):
        ot = obs[:steps, i].numpy()
        ax.plot(ot[:, 0], ot[:, 1], '--', color=colors[i], alpha=0.5, linewidth=1.5)
        ax.scatter(ot[-1, 0], ot[-1, 1], c=colors[i], s=80, edgecolors='white')
    
    rt = torch.stack(hist).numpy()
    ax.plot(rt[:, 0], rt[:, 1], 'purple', linewidth=2.5)
    ax.scatter(start[0], start[1], c='green', s=150, marker='s', zorder=10)
    ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=10)
    
    # 预测
    ax2 = axes[1]
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_aspect('equal')
    ax2.set_title('Obstacle Prediction (Upper Diffusion)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    frame = min(50, steps - 1)
    while frame > 0 and preds[frame] is None:
        frame -= 1
    
    if preds[frame] is not None:
        p = preds[frame].numpy()
        for i in range(n):
            end = min(frame + 12, obs.shape[0])
            real = obs[frame:end, i].numpy()
            ax2.plot(real[:, 0], real[:, 1], '--', color=colors[i], linewidth=2, alpha=0.5)
            ax2.plot(p[:, i, 0], p[:, i, 1], '-', color=colors[i], linewidth=2.5)
            ax2.scatter(obs[frame, i, 0], obs[frame, i, 1], c=colors[i], s=100, edgecolors='white')
    
    ax2.text(0.02, 0.98, f'Frame: {frame}', transform=ax2.transAxes, va='top')
    
    # 规划
    ax3 = axes[2]
    ax3.set_xlim(-0.8, 0.8)
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_aspect('equal')
    ax3.set_title('Trajectory Planning (Lower Diffusion)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    if trajs[frame] is not None:
        plan = trajs[frame].numpy()
        ax3.plot(plan[:, 0], plan[:, 1], 'purple', linewidth=3)
        
        for i in range(n):
            pos = obs[frame, i]
            c = Circle((pos[0], pos[1]), 0.07, color=colors[i], alpha=0.4)
            ax3.add_patch(c)
    
    ax3.scatter(hist[frame][0], hist[frame][1], c='purple', s=150, marker='s', zorder=10)
    ax3.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=10)
    
    fig.suptitle('Dual Diffusion for Dynamic Obstacle Avoidance', fontsize=14, fontweight='bold')
    
    out = '/home/wujiahao/mpd-build/dynamic_mpd/results/dual_diffusion_v2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")
    
    # 动画
    fig, ax = plt.subplots(figsize=(9, 9))
    
    def update(f):
        ax.clear()
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Frame {f}/{steps-1}', fontsize=13, fontweight='bold')
        
        for i in range(n):
            if f > 0:
                ot = obs[:f+1, i].numpy()
                ax.plot(ot[:, 0], ot[:, 1], '--', color=colors[i], alpha=0.3, linewidth=1)
            
            pos = obs[f, i]
            c = Circle((pos[0], pos[1]), 0.05, color=colors[i], alpha=0.6)
            ax.add_patch(c)
            
            if f < len(preds) and preds[f] is not None:
                p = preds[f][:, i].numpy()
                ax.plot(p[:, 0], p[:, 1], '-', color=colors[i], linewidth=2, alpha=0.5)
        
        if f > 0:
            rt = torch.stack(hist[:f+1]).numpy()
            ax.plot(rt[:, 0], rt[:, 1], 'purple', linewidth=2)
        
        ax.scatter(hist[f][0], hist[f][1], c='purple', s=180, marker='s', 
                  edgecolors='white', linewidths=2, zorder=10)
        
        if f < len(trajs) and trajs[f] is not None:
            plan = trajs[f].numpy()
            ax.plot(plan[:, 0], plan[:, 1], 'purple', linewidth=1.5, alpha=0.3, linestyle='--')
        
        ax.scatter(start[0], start[1], c='green', s=120, marker='s', zorder=5)
        ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', zorder=5)
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=80)
    out2 = '/home/wujiahao/mpd-build/dynamic_mpd/results/dual_diffusion_v2.gif'
    anim.save(out2, writer='pillow', fps=12)
    plt.close()
    print(f"  Saved: {out2}")


if __name__ == '__main__':
    run_demo()

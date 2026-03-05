"""
双Diffusion避障演示 - 完整版
正确集成原论文预训练的轨迹规划模型

关键：在导入mpd之前，patch掉isaacgym的torch导入检查
"""
import sys
import os

# ============ 关键：在导入任何东西之前，patch isaacgym的检查 ============
MPD_ROOT = '/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public'
sys.path.insert(0, MPD_ROOT)
sys.path.insert(0, os.path.join(MPD_ROOT, 'deps/isaacgym/python'))

# Patch gymdeps._import_deps 使其不检查torch
import isaacgym.gymdeps as gymdeps
_original_import_deps = gymdeps._import_deps
def _patched_import_deps():
    # 跳过torch检查，直接返回
    pass
gymdeps._import_deps = _patched_import_deps

# 现在可以安全导入torch和其他模块
sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# 导入MPD依赖
from dotmap import DotMap
from mpd.utils.loaders import load_params_from_yaml
from torch_robotics.torch_utils.torch_utils import freeze_torch_model_params

# 上层模型
from src.trainable_obstacle_diffusion import TrainableObstacleDiffusion, DiffusionConfig
from src.complex_obstacle_data import ComplexObstacleDataGenerator, ObstacleMotionConfig

print("[OK] All dependencies loaded successfully!")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def to_torch(x, device='cpu', dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class UpperObstacleDiffusion:
    """上层: 障碍物运动预测扩散模型 (我们训练的)"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.config = None
        
    def load(self, path='/home/wujiahao/mpd-build/dynamic_mpd/results/trained_diffusion_model.pth'):
        print(f"  [Upper] Loading: {os.path.basename(path)}")
        ckpt = torch.load(path, map_location=self.device)
        
        self.config = ckpt['config']
        self.model = TrainableObstacleDiffusion(self.config, self.device)
        self.model.denoise_net.load_state_dict(ckpt['model_state_dict'])
        self.model.denoise_net.eval()
        
        for p in self.model.denoise_net.parameters():
            p.requires_grad = False
        
        print(f"    Epochs: {ckpt['num_epochs']}, Best loss: {ckpt['best_test_loss']:.4f}")
        return self
    
    def predict(self, obs_history, num_samples=15):
        with torch.no_grad():
            pred_mean, pred_std, samples = self.model.predict(obs_history, num_samples)
        return pred_mean, pred_std, samples


class LowerTrajectoryDiffusion:
    """下层: 原论文预训练的轨迹规划扩散模型"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tensor_args = {'device': device, 'dtype': torch.float32}
        self.model = None
        self.args_train = None
        self.args_inference = None
        self.loaded = False
        # 模型参数 (从加载的模型获取)
        self.n_support_points = None
        self.state_dim = None
        self.context_dim = 32  # 从模型目录名
        
    def load(self, env_type='EnvSimple2D'):
        print(f"  [Lower] Loading MPD model for {env_type}...")
        
        # 加载推理配置
        cfg_path = os.path.join(MPD_ROOT, f'scripts/inference/cfgs/config_{env_type}-RobotPointMass2D_00.yaml')
        
        if not os.path.exists(cfg_path):
            print(f"    [ERROR] Config not found: {cfg_path}")
            return self
        
        self.args_inference = DotMap(load_params_from_yaml(cfg_path))
        
        # 获取模型路径
        model_dir = os.path.expandvars(self.args_inference.model_dir_ddpm_bspline)
        model_path = os.path.join(model_dir, 'checkpoints', 'ema_model_current.pth')
        
        if not os.path.exists(model_path):
            print(f"    [ERROR] Model not found: {model_path}")
            return self
        
        print(f"    Model: {model_path}")
        
        # 加载模型
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        freeze_torch_model_params(self.model)
        
        # 关键修复：将 DataParallel 替换为底层模块
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'module'):
            print("    [FIX] Unwrapping DataParallel wrapper...")
            self.model.model = self.model.model.module
        
        # 获取模型参数
        self.n_support_points = self.model.model.n_support_points  # 16
        self.state_dim = self.model.state_dim  # 2
        
        # 加载训练参数
        args_train_path = os.path.join(model_dir, 'args.yaml')
        if os.path.exists(args_train_path):
            self.args_train = DotMap(load_params_from_yaml(args_train_path))
            
        self.loaded = True
        print(f"    [OK] Loaded! Diffusion steps: {self.model.n_diffusion_steps}")
        print(f"    n_support_points: {self.n_support_points}, state_dim: {self.state_dim}")
        
        return self
    
    def plan_trajectory(self, start, goal, obstacle_positions=None, num_samples=30):
        """规划避障轨迹"""
        start = to_torch(start, **self.tensor_args)
        goal = to_torch(goal, **self.tensor_args)
        
        if self.loaded and self.model is not None:
            trajs, best = self._plan_with_mpd(start, goal, obstacle_positions, num_samples)
        else:
            trajs, best = self._plan_sampling(start, goal, obstacle_positions, num_samples)
            
        return trajs, best
    
    def _plan_with_mpd(self, start, goal, obstacle_positions, num_samples):
        """使用原论文MPD模型规划 - 带障碍物引导的版本
        
        关键改进：在扩散采样过程中加入障碍物梯度引导(Classifier-Free Guidance风格)
        使生成的轨迹主动避开预测的障碍物位置
        """
        H = self.n_support_points  # 16
        
        # 准备障碍物张量
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            obs_tensor = torch.stack([to_torch(o, **self.tensor_args) for o in obstacle_positions])
            obs_tensor_normalized = obs_tensor / 0.6  # 归一化
        else:
            obs_tensor = None
            obs_tensor_normalized = None
        
        with torch.no_grad():
            # 1. 构建 context embedding (起点+终点)
            start_normalized = start / 0.6
            goal_normalized = goal / 0.6
            
            qs_normalized = torch.cat([start_normalized, goal_normalized]).unsqueeze(0).expand(num_samples, -1)
            context = self.model.context_model(qs_normalized=qs_normalized)
            
            # 2. 设置硬约束 (起点和终点)
            hard_conds = {
                0: start_normalized.unsqueeze(0).expand(num_samples, -1),
                H - 1: goal_normalized.unsqueeze(0).expand(num_samples, -1)
            }
            
            # 3. 从噪声开始采样
            x = torch.randn(num_samples, H, self.state_dim, device=self.device)
            
            # 应用硬约束
            for idx, val in hard_conds.items():
                x[:, idx, :] = val
            
            # 障碍物引导参数 - 平衡强度和稳定性
            guidance_scale = 0.25  # 引导强度
            obs_radius_normalized = 0.12 / 0.6  # 归一化的障碍物半径
            
            # 4. DDPM 逆向扩散采样 + 障碍物引导
            for t in reversed(range(self.model.n_diffusion_steps)):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                
                # 预测噪声
                eps_pred = self.model.model(x, t_batch, context=context)
                
                # DDPM 更新参数
                alpha = self.model.alphas_cumprod[t]
                alpha_prev = self.model.alphas_cumprod_prev[t]
                beta = self.model.betas[t]
                
                # 从 epsilon 预测 x0
                x0_pred = (x - torch.sqrt(1 - alpha) * eps_pred) / torch.sqrt(alpha)
                
                # clamp防止数值爆炸
                x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
                
                # ========== 障碍物引导 ==========
                if obs_tensor_normalized is not None and t < 60:
                    guidance = torch.zeros_like(x0_pred)
                    
                    for obs_pos in obs_tensor_normalized:
                        diff = x0_pred - obs_pos.view(1, 1, 2)
                        dist = torch.norm(diff, dim=-1, keepdim=True)
                        
                        safety_margin = obs_radius_normalized * 2.0
                        mask = (dist < safety_margin).float()
                        
                        # 排斥方向
                        repulsion_dir = diff / (dist + 1e-4)
                        
                        # 排斥力
                        repulsion_strength = mask * (safety_margin - dist) / safety_margin
                        
                        guidance += repulsion_dir * repulsion_strength
                    
                    # 限制guidance幅度
                    guidance = torch.clamp(guidance, -1.0, 1.0)
                    
                    t_weight = (60 - t) / 60.0
                    x0_pred = x0_pred + guidance_scale * t_weight * guidance
                # ========== 障碍物引导结束 ==========
                
                # 计算 posterior 均值
                mean = (torch.sqrt(alpha_prev) * beta / (1 - alpha) * x0_pred 
                        + torch.sqrt(alpha) * (1 - alpha_prev) / (1 - alpha) * x)
                
                # 添加噪声 (除了 t=0)
                if t > 0:
                    noise = torch.randn_like(x)
                    variance = self.model.posterior_variance[t]
                    x = mean + torch.sqrt(variance) * noise
                else:
                    x = mean
                
                # 重新应用硬约束
                for idx, val in hard_conds.items():
                    x[:, idx, :] = val
            
            # 5. 反归一化
            trajs = x * 0.6
        
        # 6. 选择最佳轨迹（二次筛选）
        best = self._select_best(trajs, goal, obstacle_positions)
        return trajs, best
    
    def _plan_sampling(self, start, goal, obstacle_positions, num_samples):
        """基于采样的轨迹规划 (备用)"""
        H = 16
        trajs = []
        
        for _ in range(num_samples):
            t = torch.linspace(0, 1, H, device=self.device)
            base = start + t.unsqueeze(1) * (goal - start)
            noise = torch.randn(H, 2, device=self.device) * 0.08
            noise[0] = 0
            noise[-1] = 0
            traj = base + noise
            trajs.append(traj)
        
        trajs = torch.stack(trajs)
        best = self._select_best(trajs, goal, obstacle_positions)
        return trajs, best
    
    def _select_best(self, trajs, goal, obstacles, obs_radius=0.10):
        """选择最佳轨迹：避开障碍、到达目标、平滑"""
        scores = []
        
        for traj in trajs:
            # 路径长度
            lengths = torch.norm(traj[1:] - traj[:-1], dim=1)
            path_len = lengths.sum().item()
            
            # 到达目标的距离
            goal_dist = torch.norm(traj[-1] - goal).item()
            
            # 向目标方向的进展（正向运动）
            start_to_goal = goal - traj[0]
            progress = torch.dot(traj[-1] - traj[0], start_to_goal).item()
            progress_score = -progress  # 负值因为我们要最小化
            
            # 碰撞惩罚 - 大幅增加权重，使用更大的安全距离
            collision = 0
            min_clearance = float('inf')  # 最小安全距离
            if obstacles is not None:
                for obs in obstacles:
                    obs = to_torch(obs, **self.tensor_args)
                    dists = torch.norm(traj - obs, dim=1)
                    min_dist = dists.min().item()
                    min_clearance = min(min_clearance, min_dist)
                    # 软碰撞惩罚：安全距离内递增
                    collision += torch.sum(torch.relu(obs_radius * 1.5 - dists) ** 2).item()
                    # 硬碰撞惩罚：实际碰撞
                    if min_dist < obs_radius:
                        collision += 100.0  # 大惩罚
            
            # 平滑度
            if len(traj) > 2:
                v1 = traj[1:-1] - traj[:-2]
                v2 = traj[2:] - traj[1:-1]
                turns = torch.sum(torch.abs(v1 - v2)).item()
            else:
                turns = 0
            
            # 调整权重：避障最重要，其次到达目标
            score = goal_dist * 3.0 + progress_score + path_len * 0.3 + collision * 10.0 + turns * 0.1
            scores.append(score)
        
        best_idx = np.argmin(scores)
        return trajs[best_idx]


def main():
    print("\n" + "=" * 70)
    print("  Dual Diffusion Demo - Full MPD Integration")
    print("  Upper: Trainable Obstacle Prediction")
    print("  Lower: Pretrained MPD Trajectory Planning")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    max_steps = 300  # 更长的仿真
    n_obs = 6  # 更多障碍物
    
    # 加载模型
    print("\n[1/4] Loading models...")
    upper = UpperObstacleDiffusion(device).load()
    lower = LowerTrajectoryDiffusion(device).load()
    
    # 生成障碍物轨迹
    print("\n[2/4] Generating obstacles...")
    config = ObstacleMotionConfig(
        arena_size=0.6,
        num_obstacles=n_obs,
        obs_history_len=8,
        pred_horizon=12
    )
    gen = ComplexObstacleDataGenerator(config)
    
    # 为每个障碍物生成不同类型的运动轨迹
    motion_types = ['linear', 'circular', 'zigzag', 'random_walk', 'oscillating']
    trajs_list = []
    for i in range(n_obs):
        motion = motion_types[i % len(motion_types)]
        # 如果类型不支持，使用线性
        try:
            traj = gen.generate_single_obstacle_trajectory(motion, total_steps=max_steps + 100)
        except:
            traj = gen.generate_single_obstacle_trajectory('linear', total_steps=max_steps + 100)
        trajs_list.append(traj)
    
    # 堆叠为 (timesteps, n_obstacles, 2)
    obs_trajs = torch.stack(trajs_list, dim=1).to(device)
    print(f"  Obstacle trajectories shape: {obs_trajs.shape}")
    
    start = torch.tensor([-0.4, -0.4], device=device)
    goal = torch.tensor([0.4, 0.4], device=device)
    
    print(f"  Start: {start.tolist()}")
    print(f"  Goal:  {goal.tolist()}")
    print(f"  Obstacles: {n_obs}")
    
    # 仿真
    print("\n[3/4] Running simulation...")
    
    pos = start.clone()
    hist = [pos.clone()]
    obs_buf = []
    pred_hist = []
    traj_hist = []
    
    obs_len = upper.config.obs_history_len
    
    # 用于平滑控制的变量
    last_direction = (goal - start) / torch.norm(goal - start)  # 初始方向指向目标
    direction_smoothing = 0.6  # 方向平滑系数
    replan_interval = 5  # 每5步重新规划一次
    cached_traj = None
    traj_progress = 0  # 当前在轨迹上的进度
    
    for step in range(max_steps):
        obs_buf.append(obs_trajs[step].clone())
        if len(obs_buf) > obs_len:
            obs_buf.pop(0)
        
        if len(obs_buf) < 4:
            hist.append(pos.clone())
            pred_hist.append(None)
            traj_hist.append(None)
            continue
        
        # 上层: 预测障碍物
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
        
        # 构建当前和未来障碍物位置
        current_obs = [obs_trajs[step, i] for i in range(n_obs)]
        future_obs = []
        for ti in [0, 4, 8, 11]:
            for i in range(n_obs):
                future_obs.append(preds[ti, i])
        
        # 定期重新规划（纯粹依赖扩散模型的避障能力）
        # 当轨迹用完或到达重规划间隔时重新规划
        need_replan = (step % replan_interval == 0) or (cached_traj is None) or (traj_progress >= len(cached_traj) - 2)
        
        # 下层: 规划轨迹（带障碍物引导的MPD扩散模型）
        if need_replan:
            _, best = lower.plan_trajectory(pos, goal, future_obs + current_obs, 30)
            cached_traj = best
            traj_progress = 0
        
        traj_hist.append(cached_traj.clone() if cached_traj is not None else None)
        
        # 计算到目标的距离
        dist_to_goal = torch.norm(pos - goal).item()
        
        # 执行控制：跟随扩散模型规划的轨迹
        if dist_to_goal < 0.15:
            # 接近目标：增加直接向目标移动的权重
            to_goal = goal - pos
            to_goal_norm = torch.norm(to_goal)
            
            if cached_traj is not None and len(cached_traj) >= 2 and traj_progress < len(cached_traj) - 1:
                # 混合轨迹方向和目标方向
                lookahead = min(traj_progress + 2, len(cached_traj) - 1)
                target_point = cached_traj[lookahead]
                traj_direction = target_point - pos
                traj_dir_norm = torch.norm(traj_direction)
                if traj_dir_norm > 1e-6:
                    traj_direction = traj_direction / traj_dir_norm
                
                # 距离越近，越倾向于直接朝目标走
                goal_weight = max(0.5, 1.0 - dist_to_goal / 0.15)
                direction = (1 - goal_weight) * traj_direction + goal_weight * (to_goal / to_goal_norm)
                direction = direction / (torch.norm(direction) + 1e-6)
            else:
                # 没有有效轨迹点，直接朝目标
                direction = to_goal / (to_goal_norm + 1e-6)
            
            # 方向平滑
            direction = direction_smoothing * last_direction + (1 - direction_smoothing) * direction
            direction = direction / (torch.norm(direction) + 1e-6)
            last_direction = direction.clone()
            
            # 速度：接近目标时稍微减速
            speed = 0.025 if dist_to_goal > 0.08 else 0.015
            direction = direction * speed
            traj_progress += 1
        elif cached_traj is not None and len(cached_traj) >= 2 and traj_progress < len(cached_traj) - 1:
            # 正常跟随MPD规划的轨迹
            lookahead = min(traj_progress + 2, len(cached_traj) - 1)
            target_point = cached_traj[lookahead]
            
            # 计算目标方向
            target_direction = target_point - pos
            target_dir_norm = torch.norm(target_direction)
            if target_dir_norm > 1e-6:
                target_direction = target_direction / target_dir_norm
            
            # 方向平滑（控制层面，减少抖动）
            direction = direction_smoothing * last_direction + (1 - direction_smoothing) * target_direction
            direction = direction / (torch.norm(direction) + 1e-6)
            
            # 更新上一次方向
            last_direction = direction.clone()
            
            # 应用速度
            max_spd = 0.025
            direction = direction * max_spd
            
            # 更新轨迹进度
            traj_progress += 1
        else:
            # 轨迹用完或没有有效轨迹：直接朝目标
            to_goal = goal - pos
            to_goal_norm = torch.norm(to_goal)
            if to_goal_norm > 1e-6:
                direction = to_goal / to_goal_norm * 0.025
            else:
                direction = torch.zeros(2, device=device)
        
        pos = pos + direction
        
        hist.append(pos.clone())
        
        dist = torch.norm(pos - goal).item()
        if dist < 0.05:
            print(f"  [Step {step}] GOAL REACHED!")
            break
        
        if step % 20 == 0:
            print(f"  [Step {step}] Distance: {dist:.3f}")
    
    total = len(hist)
    final_dist = torch.norm(hist[-1] - goal).item()
    print(f"\n  Simulation complete: {total} steps, final dist: {final_dist:.3f}")
    
    # 可视化
    print("\n[4/4] Creating visualization...")
    visualize(hist, obs_trajs, pred_hist, traj_hist, start, goal, total, n_obs)
    
    print("\n" + "=" * 70)
    print(f"  RESULT: {'SUCCESS' if final_dist < 0.1 else 'NEED MORE STEPS'}")
    print("=" * 70)


def visualize(hist, obs, preds, trajs, start, goal, steps, n):
    """创建可视化"""
    colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12', '#1abc9c']
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    
    # 图1: 完整轨迹
    ax = axes[0]
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.set_title('Complete Trajectories', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for i in range(n):
        ot = obs[:steps, i].cpu().numpy()
        ax.plot(ot[:, 0], ot[:, 1], '--', color=colors[i], alpha=0.5, label=f'Obs {i+1}')
        ax.scatter(ot[-1, 0], ot[-1, 1], c=colors[i], s=60, edgecolors='white')
    
    rt = torch.stack(hist).cpu().numpy()
    ax.plot(rt[:, 0], rt[:, 1], 'purple', linewidth=2, label='Robot')
    ax.scatter(*start.cpu(), c='green', s=120, marker='s', zorder=10, label='Start')
    ax.scatter(*goal.cpu(), c='gold', s=180, marker='*', zorder=10, label='Goal')
    ax.legend(loc='lower left', fontsize=8)
    
    # 图2: 障碍物预测
    ax2 = axes[1]
    ax2.set_xlim(-0.7, 0.7)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_aspect('equal')
    ax2.set_title('Upper: Obstacle Prediction', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    f = min(40, len(preds) - 1, steps - 1)
    while f > 0 and (f >= len(preds) or preds[f] is None):
        f -= 1
    
    if f >= 0 and f < len(preds) and preds[f] is not None:
        p = preds[f].cpu().numpy()
        for i in range(n):
            end = min(f + 12, obs.shape[0])
            real = obs[f:end, i].cpu().numpy()
            ax2.plot(real[:, 0], real[:, 1], '--', color=colors[i], linewidth=2, alpha=0.5, label='Ground Truth' if i == 0 else None)
            ax2.plot(p[:, i, 0], p[:, i, 1], '-', color=colors[i], linewidth=2, label='Prediction' if i == 0 else None)
            ax2.scatter(obs[f, i, 0].cpu(), obs[f, i, 1].cpu(), c=colors[i], s=80, edgecolors='white')
    
    ax2.legend(loc='lower left', fontsize=8)
    ax2.text(0.02, 0.98, f'Frame: {f}', transform=ax2.transAxes, va='top')
    
    # 图3: 轨迹规划
    ax3 = axes[2]
    ax3.set_xlim(-0.7, 0.7)
    ax3.set_ylim(-0.7, 0.7)
    ax3.set_aspect('equal')
    ax3.set_title('Lower: Trajectory Planning (MPD)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    if f < len(trajs) and trajs[f] is not None:
        plan = trajs[f].cpu().numpy()
        ax3.plot(plan[:, 0], plan[:, 1], 'purple', linewidth=2.5, label='Planned Path')
        for i in range(n):
            c = Circle((obs[f, i, 0].cpu(), obs[f, i, 1].cpu()), 0.10, color=colors[i], alpha=0.4)
            ax3.add_patch(c)
    
    if f < len(hist):
        ax3.scatter(hist[f][0].cpu(), hist[f][1].cpu(), c='purple', s=120, marker='s', zorder=10)
    ax3.scatter(*goal.cpu(), c='gold', s=180, marker='*', zorder=10)
    ax3.legend(loc='lower left', fontsize=8)
    
    fig.suptitle('Dual Diffusion: Upper (Prediction) + Lower (MPD Planning)', fontsize=13, fontweight='bold')
    
    out = '/home/wujiahao/mpd-build/dynamic_mpd/results/dual_full.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")
    
    # 动画
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(f):
        ax.clear()
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Dual Diffusion | Frame {f}/{steps-1}', fontweight='bold')
        
        for i in range(n):
            pos_obs = obs[f, i].cpu()
            c = Circle((pos_obs[0], pos_obs[1]), 0.08, color=colors[i], alpha=0.6)
            ax.add_patch(c)
            if f < len(preds) and preds[f] is not None:
                p = preds[f][:, i].cpu().numpy()
                ax.plot(p[:, 0], p[:, 1], '-', color=colors[i], alpha=0.5, linewidth=2)
        
        if f > 0 and f < len(hist):
            rt = torch.stack(hist[:f+1]).cpu().numpy()
            ax.plot(rt[:, 0], rt[:, 1], 'purple', linewidth=2)
        
        if f < len(hist):
            ax.scatter(hist[f][0].cpu(), hist[f][1].cpu(), c='purple', s=150, marker='s', edgecolors='white', zorder=10)
        
        if f < len(trajs) and trajs[f] is not None:
            plan = trajs[f].cpu().numpy()
            ax.plot(plan[:, 0], plan[:, 1], 'purple', linestyle='--', alpha=0.3)
        
        ax.scatter(*start.cpu(), c='green', s=100, marker='s', zorder=5)
        ax.scatter(*goal.cpu(), c='gold', s=150, marker='*', zorder=5)
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=80)
    out2 = '/home/wujiahao/mpd-build/dynamic_mpd/results/dual_full.gif'
    anim.save(out2, writer='pillow', fps=12)
    plt.close()
    print(f"  Saved: {out2}")


if __name__ == '__main__':
    main()

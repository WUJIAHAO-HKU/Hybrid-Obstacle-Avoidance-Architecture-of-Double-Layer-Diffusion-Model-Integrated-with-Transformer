"""
扩散模型收敛过程可视化 - 增强版
用于论文展示：上层障碍物预测 + 下层轨迹规划的去噪过程

生成：
1. 上层扩散模型去噪过程图 (多帧收敛 + 不确定性)
2. 下层MPD扩散模型去噪过程图 (多帧收敛 + 多样本)
3. 双层联合收敛GIF动画
4. 噪声水平变化图
5. 论文级综合图
"""
import sys
import os

# ============ 路径设置 ============
MPD_ROOT = '/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public'
sys.path.insert(0, MPD_ROOT)
sys.path.insert(0, os.path.join(MPD_ROOT, 'deps/isaacgym/python'))

import isaacgym.gymdeps as gymdeps
gymdeps._import_deps = lambda: None

sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dotmap import DotMap
from mpd.utils.loaders import load_params_from_yaml
from torch_robotics.torch_utils.torch_utils import freeze_torch_model_params

from src.trainable_obstacle_diffusion import TrainableObstacleDiffusion, DiffusionConfig
from src.complex_obstacle_data import ComplexObstacleDataGenerator, ObstacleMotionConfig

print("[OK] Dependencies loaded!")

# 设置随机种子，确保可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
print(f"[OK] Random seed set to {SEED}")

# 设置matplotlib - 论文级别
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150


def to_torch(x, device='cpu', dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class UpperDiffusionWithProcess:
    """上层扩散模型 - 带过程记录"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.config = None
        
    def load(self, path='/home/wujiahao/mpd-build/dynamic_mpd/results/trained_diffusion_model.pth'):
        ckpt = torch.load(path, map_location=self.device)
        self.config = ckpt['config']
        self.model = TrainableObstacleDiffusion(self.config, self.device)
        self.model.denoise_net.load_state_dict(ckpt['model_state_dict'])
        self.model.denoise_net.eval()
        for p in self.model.denoise_net.parameters():
            p.requires_grad = False
        print(f"  [Upper] Loaded: epochs={ckpt['num_epochs']}, loss={ckpt['best_test_loss']:.4f}")
        return self
    
    def predict_with_process(self, obs_history, num_samples=1, record_steps=None):
        """预测并记录去噪过程"""
        if record_steps is None:
            record_steps = [0, 10, 25, 50, 75, 99]  # 记录的时间步
        
        history = obs_history.unsqueeze(0).expand(num_samples, -1, -1).to(self.device)
        T = self.config.diffusion_steps
        pred_len = self.config.pred_horizon
        
        # 从纯噪声开始
        x = torch.randn(num_samples, pred_len, 2, device=self.device)
        
        process_snapshots = []
        
        with torch.no_grad():
            for t in reversed(range(T)):
                t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                
                # 记录特定步骤
                if t in record_steps or t == T-1:
                    process_snapshots.append({
                        'step': T - 1 - t,  # 转换为正向步数
                        't': t,
                        'x': x.clone().cpu()
                    })
                
                # 预测噪声
                noise_pred = self.model.denoise_net(x, history, t_tensor)
                
                # 去噪 - 使用正确的属性名
                alpha_bar = self.model.alpha_bars[t]
                alpha_bar_prev = self.model.alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=self.device)
                beta = self.model.betas[t]
                alpha = self.model.alphas[t]
                
                x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
                
                if t > 0:
                    # 计算posterior均值和方差
                    posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    mean = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar) * x0_pred +
                            torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar) * x)
                    x = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
                else:
                    x = x0_pred
            
            # 最终结果
            process_snapshots.append({
                'step': T,
                't': -1,
                'x': x.clone().cpu()
            })
        
        return x.mean(dim=0).cpu(), process_snapshots


class LowerDiffusionWithProcess:
    """下层MPD扩散模型 - 带过程记录"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tensor_args = {'device': device, 'dtype': torch.float32}
        self.model = None
        self.loaded = False
        
    def load(self, env_type='EnvSimple2D'):
        cfg_path = os.path.join(MPD_ROOT, f'scripts/inference/cfgs/config_{env_type}-RobotPointMass2D_00.yaml')
        args_inference = DotMap(load_params_from_yaml(cfg_path))
        model_dir = os.path.expandvars(args_inference.model_dir_ddpm_bspline)
        model_path = os.path.join(model_dir, 'checkpoints', 'ema_model_current.pth')
        
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        freeze_torch_model_params(self.model)
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'module'):
            self.model.model = self.model.model.module
        
        self.n_support_points = self.model.model.n_support_points
        self.state_dim = self.model.state_dim
        self.loaded = True
        print(f"  [Lower] Loaded: diffusion_steps={self.model.n_diffusion_steps}, H={self.n_support_points}")
        return self
    
    def plan_with_process(self, start, goal, obstacles=None, num_samples=1, record_steps=None):
        """规划轨迹并记录去噪过程"""
        if record_steps is None:
            record_steps = [0, 10, 25, 50, 75, 99]
        
        H = self.n_support_points
        T = self.model.n_diffusion_steps
        
        # 准备障碍物
        if obstacles is not None and len(obstacles) > 0:
            obs_tensor = torch.stack([to_torch(o, **self.tensor_args) for o in obstacles])
            obs_normalized = obs_tensor / 0.6
        else:
            obs_normalized = None
        
        start_n = to_torch(start, **self.tensor_args) / 0.6
        goal_n = to_torch(goal, **self.tensor_args) / 0.6
        
        process_snapshots = []
        
        with torch.no_grad():
            qs_normalized = torch.cat([start_n, goal_n]).unsqueeze(0).expand(num_samples, -1)
            context = self.model.context_model(qs_normalized=qs_normalized)
            
            hard_conds = {
                0: start_n.unsqueeze(0).expand(num_samples, -1),
                H - 1: goal_n.unsqueeze(0).expand(num_samples, -1)
            }
            
            # 从噪声开始
            x = torch.randn(num_samples, H, self.state_dim, device=self.device)
            for idx, val in hard_conds.items():
                x[:, idx, :] = val
            
            guidance_scale = 0.25
            obs_radius_n = 0.12 / 0.6
            
            for t in reversed(range(T)):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                
                # 记录
                if t in record_steps or t == T-1:
                    process_snapshots.append({
                        'step': T - 1 - t,
                        't': t,
                        'x': (x.clone() * 0.6).cpu()  # 反归一化
                    })
                
                eps_pred = self.model.model(x, t_batch, context=context)
                
                alpha = self.model.alphas_cumprod[t]
                alpha_prev = self.model.alphas_cumprod_prev[t]
                beta = self.model.betas[t]
                
                x0_pred = (x - torch.sqrt(1 - alpha) * eps_pred) / torch.sqrt(alpha)
                x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
                
                # 障碍物引导
                if obs_normalized is not None and t < 60:
                    guidance = torch.zeros_like(x0_pred)
                    for obs_pos in obs_normalized:
                        diff = x0_pred - obs_pos.view(1, 1, 2)
                        dist = torch.norm(diff, dim=-1, keepdim=True)
                        safety = obs_radius_n * 2.0
                        mask = (dist < safety).float()
                        rep_dir = diff / (dist + 1e-4)
                        rep_str = mask * (safety - dist) / safety
                        guidance += rep_dir * rep_str
                    guidance = torch.clamp(guidance, -1.0, 1.0)
                    t_weight = (60 - t) / 60.0
                    x0_pred = x0_pred + guidance_scale * t_weight * guidance
                
                mean = (torch.sqrt(alpha_prev) * beta / (1 - alpha) * x0_pred +
                        torch.sqrt(alpha) * (1 - alpha_prev) / (1 - alpha) * x)
                
                if t > 0:
                    variance = self.model.posterior_variance[t]
                    x = mean + torch.sqrt(variance) * torch.randn_like(x)
                else:
                    x = mean
                
                for idx, val in hard_conds.items():
                    x[:, idx, :] = val
            
            # 最终结果
            process_snapshots.append({
                'step': T,
                't': -1,
                'x': (x.clone() * 0.6).cpu()
            })
        
        return (x * 0.6).cpu(), process_snapshots


def compute_prediction_error(pred, gt):
    """计算预测误差 (MSE)"""
    pred = pred if isinstance(pred, np.ndarray) else pred.numpy()
    gt = gt if isinstance(gt, np.ndarray) else gt.cpu().numpy()
    min_len = min(len(pred), len(gt))
    return np.mean((pred[:min_len] - gt[:min_len]) ** 2)


def compute_trajectory_smoothness(traj):
    """计算轨迹平滑度 (曲率变化)"""
    traj = traj if isinstance(traj, np.ndarray) else traj.numpy()
    if len(traj) < 3:
        return 0
    v1 = traj[1:-1] - traj[:-2]
    v2 = traj[2:] - traj[1:-1]
    # 角度变化
    angles = []
    for i in range(len(v1)):
        cos_angle = np.dot(v1[i], v2[i]) / (np.linalg.norm(v1[i]) * np.linalg.norm(v2[i]) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angles.append(np.arccos(cos_angle))
    return np.mean(angles) if angles else 0


def create_upper_diffusion_figure(snapshots, obs_history, ground_truth, save_path):
    """创建上层扩散模型收敛过程图 - 增强版"""
    n_snapshots = len(snapshots)
    
    # 使用GridSpec实现更灵活的布局
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1.2, 0.6], hspace=0.3, wspace=0.25)
    
    # 颜色渐变
    cmap = plt.cm.viridis
    gt = ground_truth.cpu().numpy()
    hist = obs_history.cpu().numpy()
    
    # 计算每个步骤的误差
    errors = []
    smoothness_vals = []
    steps_list = []
    
    # ===== 上部两行：去噪过程快照 =====
    for idx, snap in enumerate(snapshots[:6]):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        step = snap['step']
        t = snap['t']
        x = snap['x'][0]
        pred = x.numpy()
        
        # 计算误差
        error = compute_prediction_error(pred, gt)
        smoothness = compute_trajectory_smoothness(pred)
        errors.append(error)
        smoothness_vals.append(smoothness)
        steps_list.append(step)
        
        # 绘制历史轨迹（灰色实心圆）
        ax.plot(hist[:, 0], hist[:, 1], 'o-', color='#7f8c8d', alpha=0.6, 
                markersize=5, linewidth=2, label='History' if idx == 0 else None)
        ax.scatter(hist[-1, 0], hist[-1, 1], c='#7f8c8d', s=100, zorder=5, 
                   edgecolors='black', linewidth=1.5)
        
        # 绘制当前预测
        progress = step / 50.0  # 上层模型是50步
        color = cmap(0.2 + 0.8 * min(progress, 1.0))
        
        # 绘制轨迹带不确定性（用线宽表示）
        lw = 1.5 + 2.0 * progress  # 线宽随收敛增加
        ax.plot(pred[:, 0], pred[:, 1], 'o-', color=color, 
                markersize=4 + 3 * progress, linewidth=lw, alpha=0.85, 
                label='Prediction' if idx == 0 else None)
        
        # 预测终点
        ax.scatter(pred[-1, 0], pred[-1, 1], c=color, s=120, zorder=6, 
                   edgecolors='white', linewidth=2, marker='D')
        
        # 绘制真实轨迹
        ax.plot(gt[:, 0], gt[:, 1], '--', color='#e74c3c', 
                linewidth=2, alpha=0.5, label='Ground Truth' if idx == 0 else None)
        
        # 添加噪声水平指示器（右上角进度条）
        noise_level = 1.0 - progress
        bar_x, bar_y = 0.55, 0.55
        bar_w, bar_h = 0.12, 0.03
        # 背景
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, 
                               facecolor='white', edgecolor='black', linewidth=1, zorder=7))
        # 填充
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w * noise_level, bar_h, 
                               facecolor='#3498db', alpha=0.7, zorder=8))
        ax.text(bar_x + bar_w/2, bar_y + bar_h + 0.02, f'Noise: {noise_level*100:.0f}%', 
               fontsize=7, ha='center', fontweight='bold')
        
        # 标题带误差信息
        if t >= 0:
            ax.set_title(f'Step {step}/50 (t={t})\nMSE: {error:.4f}', fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'Final Result\nMSE: {error:.4f}', fontsize=11, fontweight='bold', color='green')
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    
    # ===== 右侧：误差收敛曲线 =====
    ax_error = fig.add_subplot(gs[0:2, 3])
    ax_error.plot(steps_list, errors, 'o-', color='#e74c3c', linewidth=2.5, 
                  markersize=10, label='MSE Error')
    ax_error.fill_between(steps_list, 0, errors, color='#e74c3c', alpha=0.15)
    
    # 添加收敛趋势线
    if len(steps_list) > 2:
        z = np.polyfit(steps_list, errors, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(steps_list), max(steps_list), 50)
        ax_error.plot(x_smooth, p(x_smooth), '--', color='#9b59b6', alpha=0.7, 
                      linewidth=1.5, label='Trend')
    
    ax_error.set_xlabel('Denoising Step', fontsize=11)
    ax_error.set_ylabel('Mean Squared Error', fontsize=11)
    ax_error.set_title('Error Convergence', fontsize=12, fontweight='bold')
    ax_error.grid(True, alpha=0.3)
    ax_error.legend(fontsize=9)
    ax_error.set_ylim(bottom=0)
    
    # ===== 底部：综合信息 =====
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')
    
    # 添加信息文本
    info_text = (
        f"📊 Upper Diffusion Model Statistics\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• Diffusion Steps: 50  |  History Length: {len(hist)} frames  |  Prediction Horizon: {len(gt)} frames\n"
        f"• Initial MSE: {errors[0]:.4f}  →  Final MSE: {errors[-1]:.4f}  (↓ {(1-errors[-1]/errors[0])*100:.1f}% reduction)\n"
        f"• Smoothness (Final): {smoothness_vals[-1]:.4f} rad (avg turn angle)"
    )
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes, 
                 fontsize=11, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#bdc3c7'),
                 family='monospace')
    
    fig.suptitle('Upper Diffusion: Obstacle Trajectory Prediction Denoising Process', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_collision_risk(traj, obstacles, obs_radius=0.08):
    """计算轨迹与障碍物的碰撞风险"""
    traj = traj if isinstance(traj, np.ndarray) else traj.numpy()
    if obstacles is None:
        return 0
    
    min_dist = float('inf')
    collision_count = 0
    
    for obs in obstacles:
        obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
        for pt in traj:
            dist = np.linalg.norm(pt - obs_np)
            min_dist = min(min_dist, dist)
            if dist < obs_radius:
                collision_count += 1
    
    return min_dist, collision_count


def compute_path_length(traj):
    """计算轨迹总长度"""
    traj = traj if isinstance(traj, np.ndarray) else traj.numpy()
    lengths = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    return np.sum(lengths)


def create_lower_diffusion_figure(snapshots, start, goal, obstacles, save_path):
    """创建下层MPD扩散模型收敛过程图 - 增强版"""
    n_snapshots = len(snapshots)
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1.2, 0.6], hspace=0.3, wspace=0.25)
    
    cmap = plt.cm.plasma
    obs_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12', '#1abc9c']
    
    # 收集统计数据
    path_lengths = []
    min_clearances = []
    collision_counts = []
    steps_list = []
    smoothness_vals = []
    
    # 计算理想直线距离
    ideal_dist = np.linalg.norm(np.array(goal) - np.array(start))
    
    # ===== 上部两行：去噪过程快照 =====
    for idx, snap in enumerate(snapshots[:6]):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        step = snap['step']
        t = snap['t']
        x = snap['x'][0]
        traj = x.numpy()
        
        # 计算统计信息
        path_len = compute_path_length(traj)
        min_clear, collisions = compute_collision_risk(traj, obstacles)
        smoothness = compute_trajectory_smoothness(traj)
        
        path_lengths.append(path_len)
        min_clearances.append(min_clear if min_clear != float('inf') else 0.5)
        collision_counts.append(collisions)
        steps_list.append(step)
        smoothness_vals.append(smoothness)
        
        # 绘制障碍物（带危险区域）
        if obstacles is not None:
            for i, obs in enumerate(obstacles[:6]):
                obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
                # 危险区域（外圈）
                danger_zone = Circle((obs_np[0], obs_np[1]), 0.12, 
                                     facecolor=obs_colors[i % len(obs_colors)], 
                                     alpha=0.15, linestyle='--', edgecolor=obs_colors[i % len(obs_colors)])
                ax.add_patch(danger_zone)
                # 障碍物本体
                c = Circle((obs_np[0], obs_np[1]), 0.08, 
                          color=obs_colors[i % len(obs_colors)], alpha=0.5)
                ax.add_patch(c)
        
        # 绘制轨迹
        progress = step / 100.0
        color = cmap(0.15 + 0.85 * min(progress, 1.0))
        
        # 使用渐变线条表示轨迹
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 线宽渐变（起点粗，终点细）
        lws = np.linspace(3.5, 1.5, len(segments))
        for seg, lw in zip(segments, lws):
            ax.plot(seg[:, 0], seg[:, 1], '-', color=color, linewidth=lw, alpha=0.9)
        
        # 轨迹点（带渐变大小）
        sizes = np.linspace(50, 20, len(traj))
        ax.scatter(traj[:, 0], traj[:, 1], c=[color]*len(traj), 
                   s=sizes, zorder=4, edgecolors='white', linewidth=0.8)
        
        # 绘制起点和终点
        ax.scatter(start[0], start[1], c='#27ae60', s=180, marker='s', 
                   zorder=10, edgecolors='white', linewidth=2.5, label='Start' if idx == 0 else None)
        ax.scatter(goal[0], goal[1], c='gold', s=250, marker='*', 
                   zorder=10, edgecolors='black', linewidth=1.5, label='Goal' if idx == 0 else None)
        
        # 添加路径效率指示器
        efficiency = ideal_dist / path_len if path_len > 0 else 0
        eff_text = f"Eff: {efficiency*100:.0f}%"
        ax.text(0.02, 0.98, eff_text, transform=ax.transAxes, fontsize=9,
               va='top', ha='left', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 碰撞警告
        if collisions > 0:
            ax.text(0.98, 0.02, f'⚠️ {collisions} collisions', transform=ax.transAxes,
                   fontsize=9, va='bottom', ha='right', color='red', fontweight='bold')
        
        # 噪声水平条
        noise_level = 1.0 - progress
        bar_x, bar_y = 0.55, 0.55
        bar_w, bar_h = 0.12, 0.03
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, 
                               facecolor='white', edgecolor='black', linewidth=1, zorder=7))
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w * noise_level, bar_h, 
                               facecolor='#9b59b6', alpha=0.7, zorder=8))
        
        # 标题
        if t >= 0:
            ax.set_title(f'Step {step}/100 (t={t})\nLength: {path_len:.3f}', 
                        fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'Final Trajectory\nLength: {path_len:.3f}', 
                        fontsize=11, fontweight='bold', color='green')
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    
    # ===== 右侧：多指标收敛曲线 =====
    ax_metrics = fig.add_subplot(gs[0:2, 3])
    
    # 归一化数据
    path_norm = np.array(path_lengths) / max(path_lengths)
    clear_norm = np.array(min_clearances) / max(min_clearances) if max(min_clearances) > 0 else min_clearances
    smooth_norm = np.array(smoothness_vals) / max(smoothness_vals) if max(smoothness_vals) > 0 else smoothness_vals
    
    ax_metrics.plot(steps_list, path_norm, 'o-', color='#e74c3c', linewidth=2.5, 
                    markersize=8, label='Path Length')
    ax_metrics.plot(steps_list, clear_norm, 's-', color='#27ae60', linewidth=2.5, 
                    markersize=8, label='Min Clearance')
    ax_metrics.plot(steps_list, 1 - smooth_norm, '^-', color='#3498db', linewidth=2.5, 
                    markersize=8, label='Smoothness')
    
    ax_metrics.set_xlabel('Denoising Step', fontsize=11)
    ax_metrics.set_ylabel('Normalized Metric', fontsize=11)
    ax_metrics.set_title('Trajectory Quality Metrics', fontsize=12, fontweight='bold')
    ax_metrics.grid(True, alpha=0.3)
    ax_metrics.legend(fontsize=9, loc='best')
    ax_metrics.set_ylim(0, 1.1)
    
    # ===== 底部：综合信息 =====
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')
    
    info_text = (
        f"📊 Lower MPD Diffusion Model Statistics\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• Diffusion Steps: 100  |  Support Points: 16  |  Ideal Distance: {ideal_dist:.3f}\n"
        f"• Path Length: {path_lengths[0]:.3f} → {path_lengths[-1]:.3f}  |  "
        f"Path Efficiency: {ideal_dist/path_lengths[-1]*100:.1f}%\n"
        f"• Min Clearance: {min_clearances[0]:.3f} → {min_clearances[-1]:.3f}  |  "
        f"Collisions: {collision_counts[0]} → {collision_counts[-1]}"
    )
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes, 
                 fontsize=11, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f6f3', edgecolor='#1abc9c'),
                 family='monospace')
    
    fig.suptitle('Lower Diffusion: MPD Trajectory Planning Denoising Process', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def create_combined_animation(upper_snapshots, lower_snapshots, 
                               obs_history, ground_truth,
                               start, goal, obstacles, save_path):
    """创建双层扩散联合收敛动画 - 增强版"""
    
    n_frames = min(len(upper_snapshots), len(lower_snapshots))
    gt = ground_truth.cpu().numpy()
    hist = obs_history.cpu().numpy()
    
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.4], wspace=0.2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_progress = fig.add_subplot(gs[0, 2])
    
    obs_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12', '#1abc9c']
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        ax_progress.clear()
        
        upper_snap = upper_snapshots[frame]
        lower_snap = lower_snapshots[frame]
        step = upper_snap['step']
        progress = step / 50.0  # 上层50步
        
        # ===== 左图：上层障碍物预测 =====
        ax1.set_xlim(-0.7, 0.7)
        ax1.set_ylim(-0.7, 0.7)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        pred = upper_snap['x'][0].numpy()
        
        # 历史轨迹（带渐变透明度）
        for i in range(len(hist)-1):
            alpha = 0.3 + 0.5 * (i / len(hist))
            ax1.plot(hist[i:i+2, 0], hist[i:i+2, 1], 'o-', color='#7f8c8d', 
                    alpha=alpha, markersize=4 + 2*i/len(hist), linewidth=1.5)
        ax1.scatter(hist[-1, 0], hist[-1, 1], c='#34495e', s=100, zorder=5, 
                   edgecolors='white', linewidth=2, marker='o')
        ax1.text(hist[-1, 0]+0.05, hist[-1, 1]+0.05, 'Now', fontsize=9, fontweight='bold')
        
        # 预测轨迹
        color = plt.cm.viridis(0.3 + 0.7 * min(progress, 1.0))
        ax1.plot(pred[:, 0], pred[:, 1], 'o-', color=color, 
                markersize=6, linewidth=2.5, alpha=0.9)
        ax1.scatter(pred[-1, 0], pred[-1, 1], c=color, s=150, zorder=6, 
                   edgecolors='white', linewidth=2, marker='D')
        
        # 真实轨迹
        if frame == n_frames - 1:
            ax1.plot(gt[:, 0], gt[:, 1], '--', color='#e74c3c', 
                    linewidth=2.5, alpha=0.8, label='Ground Truth')
            ax1.legend(loc='lower left', fontsize=9)
        else:
            # 显示部分真实轨迹作为参考
            ax1.plot(gt[:, 0], gt[:, 1], '--', color='#e74c3c', 
                    linewidth=1.5, alpha=0.3)
        
        # 误差计算
        error = compute_prediction_error(pred, gt)
        
        ax1.set_title(f'Upper: Obstacle Prediction\nStep {step}/50 | MSE: {error:.4f}', 
                     fontsize=12, fontweight='bold')
        
        # ===== 右图：下层轨迹规划 =====
        ax2.set_xlim(-0.7, 0.7)
        ax2.set_ylim(-0.7, 0.7)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        traj = lower_snap['x'][0].numpy()
        
        # 障碍物（带危险区域）
        if obstacles is not None:
            for i, obs in enumerate(obstacles[:6]):
                obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
                # 危险区域
                danger = Circle((obs_np[0], obs_np[1]), 0.12, 
                               facecolor=obs_colors[i % len(obs_colors)], 
                               alpha=0.1, linestyle=':', 
                               edgecolor=obs_colors[i % len(obs_colors)], linewidth=1)
                ax2.add_patch(danger)
                # 障碍物
                c = Circle((obs_np[0], obs_np[1]), 0.08, 
                          color=obs_colors[i % len(obs_colors)], alpha=0.6)
                ax2.add_patch(c)
        
        # 轨迹（渐变色表示方向）
        color = plt.cm.plasma(0.2 + 0.8 * min(progress, 1.0))
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        colors = [plt.cm.plasma(0.2 + 0.6 * i / len(segments)) for i in range(len(segments))]
        for seg, c in zip(segments, colors):
            ax2.plot(seg[:, 0], seg[:, 1], '-', color=c, linewidth=3, alpha=0.9)
        
        ax2.scatter(traj[:, 0], traj[:, 1], c=range(len(traj)), cmap='plasma', 
                   s=40, zorder=4, edgecolors='white', linewidth=0.5)
        
        # 起点终点
        ax2.scatter(start[0], start[1], c='#27ae60', s=180, marker='s', 
                   zorder=10, edgecolors='white', linewidth=2.5)
        ax2.scatter(goal[0], goal[1], c='gold', s=250, marker='*', 
                   zorder=10, edgecolors='black', linewidth=1.5)
        
        # 路径长度和效率
        path_len = compute_path_length(traj)
        ideal_dist = np.linalg.norm(np.array(goal) - np.array(start))
        efficiency = ideal_dist / path_len * 100 if path_len > 0 else 0
        
        ax2.set_title(f'Lower: MPD Trajectory Planning\nStep {lower_snap["step"]}/100 | Eff: {efficiency:.0f}%', 
                     fontsize=12, fontweight='bold')
        
        # ===== 右侧：进度面板 =====
        ax_progress.axis('off')
        
        # 进度条背景
        bar_height = 0.6
        bar_width = 0.4
        
        # 上层进度
        ax_progress.add_patch(Rectangle((0.1, 0.65), bar_width, 0.08, 
                                        facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=2))
        ax_progress.add_patch(Rectangle((0.1, 0.65), bar_width * min(progress, 1.0), 0.08, 
                                        facecolor='#3498db', alpha=0.8))
        ax_progress.text(0.3, 0.8, 'Upper Progress', ha='center', fontsize=10, fontweight='bold')
        ax_progress.text(0.3, 0.62, f'{min(progress*100, 100):.0f}%', ha='center', fontsize=11)
        
        # 下层进度
        lower_progress = lower_snap["step"] / 100.0
        ax_progress.add_patch(Rectangle((0.1, 0.35), bar_width, 0.08, 
                                        facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=2))
        ax_progress.add_patch(Rectangle((0.1, 0.35), bar_width * min(lower_progress, 1.0), 0.08, 
                                        facecolor='#9b59b6', alpha=0.8))
        ax_progress.text(0.3, 0.5, 'Lower Progress', ha='center', fontsize=10, fontweight='bold')
        ax_progress.text(0.3, 0.32, f'{min(lower_progress*100, 100):.0f}%', ha='center', fontsize=11)
        
        # 统计信息
        info_text = f"Frame: {frame+1}/{n_frames}\n\nMSE: {error:.4f}\nPath Eff: {efficiency:.0f}%"
        ax_progress.text(0.3, 0.15, info_text, ha='center', va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_progress.set_xlim(0, 0.6)
        ax_progress.set_ylim(0, 1)
        
        fig.suptitle('Dual Diffusion Model: Denoising Convergence Animation', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=600, blit=False)
    anim.save(save_path, writer='pillow', fps=2, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def create_side_by_side_convergence(upper_snapshots, lower_snapshots,
                                     obs_history, ground_truth,
                                     start, goal, obstacles, save_path):
    """创建并排的收敛过程对比图 - 论文级增强版（无收敛曲线）"""
    
    n_steps = min(6, len(upper_snapshots), len(lower_snapshots))
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, n_steps, figure=fig, height_ratios=[1.2, 1.2, 0.5], 
                  hspace=0.25, wspace=0.15)
    
    obs_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12', '#1abc9c']
    gt = ground_truth.cpu().numpy()
    hist = obs_history.cpu().numpy()
    
    for col in range(n_steps):
        # ===== 上行：上层障碍物预测 =====
        ax_upper = fig.add_subplot(gs[0, col])
        ax_upper.set_xlim(-0.7, 0.7)
        ax_upper.set_ylim(-0.7, 0.7)
        ax_upper.set_aspect('equal')
        ax_upper.grid(True, alpha=0.2, linestyle=':')
        
        if col < len(upper_snapshots):
            snap = upper_snapshots[col]
            step = snap['step']
            pred = snap['x'][0].numpy()
            
            # 历史（灰色虚线）
            ax_upper.plot(hist[:, 0], hist[:, 1], 'o--', color='#95a5a6', 
                         alpha=0.6, markersize=3, linewidth=1)
            ax_upper.scatter(hist[-1, 0], hist[-1, 1], c='#34495e', s=60, zorder=5, 
                           edgecolors='white', linewidth=1)
            
            # 预测
            progress = step / 50.0
            color = plt.cm.viridis(0.25 + 0.75 * min(progress, 1.0))
            ax_upper.plot(pred[:, 0], pred[:, 1], 'o-', color=color, 
                         markersize=4, linewidth=2, alpha=0.85)
            ax_upper.scatter(pred[-1, 0], pred[-1, 1], c=color, s=80, zorder=6,
                           edgecolors='white', linewidth=1.5, marker='D')
            
            # 真实轨迹（最后两帧显示）
            if col >= n_steps - 2:
                ax_upper.plot(gt[:, 0], gt[:, 1], '--', color='#e74c3c', 
                             linewidth=1.8, alpha=0.7)
            
            # 噪声指示器
            noise = 1.0 - min(progress, 1.0)
            ax_upper.text(0.5, 0.95, f'σ={noise:.1f}', transform=ax_upper.transAxes,
                         fontsize=8, ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            title = f't={50-step}' if step < 50 else 'Final'
            ax_upper.set_title(title, fontsize=11, fontweight='bold')
        
        # 去掉内部坐标轴刻度
        if col > 0:
            ax_upper.set_yticklabels([])
        ax_upper.set_xticklabels([])
        
        if col == 0:
            ax_upper.set_ylabel('Upper\n(Prediction)', fontsize=12, fontweight='bold')
        
        # ===== 下行：下层轨迹规划 =====
        ax_lower = fig.add_subplot(gs[1, col])
        ax_lower.set_xlim(-0.7, 0.7)
        ax_lower.set_ylim(-0.7, 0.7)
        ax_lower.set_aspect('equal')
        ax_lower.grid(True, alpha=0.2, linestyle=':')
        
        if col < len(lower_snapshots):
            snap = lower_snapshots[col]
            step = snap['step']
            traj = snap['x'][0].numpy()
            
            path_len = compute_path_length(traj)
            
            # 障碍物
            if obstacles is not None:
                for i, obs in enumerate(obstacles[:6]):
                    obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
                    # 安全区域
                    ax_lower.add_patch(Circle((obs_np[0], obs_np[1]), 0.10, 
                                             facecolor=obs_colors[i % len(obs_colors)], 
                                             alpha=0.1, linestyle='--',
                                             edgecolor=obs_colors[i % len(obs_colors)]))
                    # 障碍物
                    ax_lower.add_patch(Circle((obs_np[0], obs_np[1]), 0.06, 
                                             color=obs_colors[i % len(obs_colors)], alpha=0.4))
            
            # 轨迹（带方向渐变）
            progress = step / 100.0
            for i in range(len(traj) - 1):
                color = plt.cm.plasma(0.15 + 0.85 * i / len(traj))
                lw = 2.5 - 1.0 * i / len(traj)  # 起点粗，终点细
                ax_lower.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', 
                             color=color, linewidth=lw, alpha=0.9)
            
            # 起点终点
            ax_lower.scatter(start[0], start[1], c='#27ae60', s=100, marker='s', 
                           zorder=10, edgecolors='white', linewidth=2)
            ax_lower.scatter(goal[0], goal[1], c='gold', s=130, marker='*', 
                           zorder=10, edgecolors='black', linewidth=1)
            
            # 路径效率
            ideal = np.linalg.norm(np.array(goal) - np.array(start))
            eff = ideal / path_len * 100 if path_len > 0 else 0
            ax_lower.text(0.5, 0.95, f'η={eff:.0f}%', transform=ax_lower.transAxes,
                         fontsize=8, ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            title = f't={100-step}' if step < 100 else 'Final'
            ax_lower.set_title(title, fontsize=11, fontweight='bold')
        
        if col > 0:
            ax_lower.set_yticklabels([])
        
        if col == 0:
            ax_lower.set_ylabel('Lower\n(Planning)', fontsize=12, fontweight='bold')
    
    # ===== 底部：说明和图例 =====
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    
    # 时间流向箭头
    arrow = FancyArrowPatch((0.15, 0.7), (0.85, 0.7), 
                            arrowstyle='->', mutation_scale=20,
                            color='#34495e', linewidth=2)
    ax_legend.add_patch(arrow)
    ax_legend.text(0.5, 0.85, 'Denoising Direction (Noise → Clean)', 
                  ha='center', fontsize=12, fontweight='bold', transform=ax_legend.transAxes)
    
    # 图例说明
    legend_text = (
        "Upper Model: Predicts future obstacle trajectories from historical observations\n"
        "Lower Model: Plans collision-free path using obstacle guidance in diffusion sampling\n"
        f"σ: Noise level | η: Path efficiency | MSE: Mean Squared Error"
    )
    ax_legend.text(0.5, 0.25, legend_text, ha='center', va='center', fontsize=10,
                  transform=ax_legend.transAxes, style='italic',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    fig.suptitle('Dual Diffusion Model: Denoising Process Comparison\n(Noise → Clean)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def create_noise_schedule_figure(save_path):
    """创建噪声调度可视化图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 上层模型噪声调度（50步）
    T_upper = 50
    beta_start, beta_end = 1e-4, 0.02
    betas_upper = np.linspace(beta_start, beta_end, T_upper)
    alphas_upper = 1 - betas_upper
    alpha_bars_upper = np.cumprod(alphas_upper)
    
    ax1 = axes[0]
    ax1.plot(range(T_upper), alpha_bars_upper, 'b-', linewidth=2.5, label=r'$\bar{\alpha}_t$')
    ax1.plot(range(T_upper), np.sqrt(alpha_bars_upper), 'g--', linewidth=2, label=r'$\sqrt{\bar{\alpha}_t}$ (Signal)')
    ax1.plot(range(T_upper), np.sqrt(1 - alpha_bars_upper), 'r:', linewidth=2, label=r'$\sqrt{1-\bar{\alpha}_t}$ (Noise)')
    ax1.fill_between(range(T_upper), 0, alpha_bars_upper, alpha=0.1, color='blue')
    ax1.set_xlabel('Diffusion Step t', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Upper Model: Noise Schedule (50 steps)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T_upper-1)
    ax1.set_ylim(0, 1.05)
    
    # 标注关键区域
    ax1.axvspan(0, 10, alpha=0.1, color='green', label='Low noise')
    ax1.axvspan(40, 50, alpha=0.1, color='red', label='High noise')
    ax1.text(5, 0.15, 'Clean\nRegion', ha='center', fontsize=9, style='italic')
    ax1.text(45, 0.15, 'Noisy\nRegion', ha='center', fontsize=9, style='italic')
    
    # 下层模型噪声调度（100步）
    T_lower = 100
    betas_lower = np.linspace(beta_start, beta_end, T_lower)
    alphas_lower = 1 - betas_lower
    alpha_bars_lower = np.cumprod(alphas_lower)
    
    ax2 = axes[1]
    ax2.plot(range(T_lower), alpha_bars_lower, 'b-', linewidth=2.5, label=r'$\bar{\alpha}_t$')
    ax2.plot(range(T_lower), np.sqrt(alpha_bars_lower), 'g--', linewidth=2, label=r'$\sqrt{\bar{\alpha}_t}$ (Signal)')
    ax2.plot(range(T_lower), np.sqrt(1 - alpha_bars_lower), 'r:', linewidth=2, label=r'$\sqrt{1-\bar{\alpha}_t}$ (Noise)')
    ax2.fill_between(range(T_lower), 0, alpha_bars_lower, alpha=0.1, color='blue')
    ax2.set_xlabel('Diffusion Step t', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Lower Model: Noise Schedule (100 steps)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, T_lower-1)
    ax2.set_ylim(0, 1.05)
    
    # 标注obstacle guidance区域
    ax2.axvspan(0, 60, alpha=0.15, color='orange')
    ax2.text(30, 0.05, 'Obstacle Guidance\nActive Region (t<60)', 
            ha='center', fontsize=9, style='italic', color='#d35400')
    
    fig.suptitle('Diffusion Noise Schedule: Signal-to-Noise Ratio Over Time', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def create_multi_sample_diversity_figure(lower_model, start, goal, obstacles, save_path):
    """创建多样本多样性可视化图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    obs_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12', '#1abc9c']
    sample_colors = plt.cm.Set2(np.linspace(0, 1, 10))
    
    # 生成多个样本
    num_samples = 10
    trajs, _ = lower_model.plan_with_process(start, goal, obstacles, 
                                              num_samples=num_samples, 
                                              record_steps=[99, 50, 0])
    
    titles = ['All Samples', 'Top 3 by Smoothness', 'Diversity Heatmap']
    
    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 障碍物
        if obstacles is not None:
            for i, obs in enumerate(obstacles[:6]):
                obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
                c = Circle((obs_np[0], obs_np[1]), 0.08, 
                          color=obs_colors[i % len(obs_colors)], alpha=0.4)
                ax.add_patch(c)
        
        if ax_idx == 0:
            # 显示所有样本
            for i in range(num_samples):
                traj = trajs[i].numpy()
                ax.plot(traj[:, 0], traj[:, 1], '-', color=sample_colors[i], 
                       linewidth=1.5, alpha=0.7)
            ax.set_title('All Samples (n=10)', fontsize=12, fontweight='bold')
            
        elif ax_idx == 1:
            # 按平滑度排序，显示前3
            smoothness_scores = []
            for i in range(num_samples):
                traj = trajs[i].numpy()
                smooth = compute_trajectory_smoothness(traj)
                smoothness_scores.append((i, smooth))
            
            smoothness_scores.sort(key=lambda x: x[1])
            top3 = [s[0] for s in smoothness_scores[:3]]
            
            for rank, i in enumerate(top3):
                traj = trajs[i].numpy()
                lw = 3.5 - rank * 0.8
                ax.plot(traj[:, 0], traj[:, 1], '-', color=sample_colors[i], 
                       linewidth=lw, alpha=0.9, label=f'Sample {i+1}')
            ax.legend(fontsize=9, loc='lower left')
            ax.set_title('Top 3 by Smoothness', fontsize=12, fontweight='bold')
            
        else:
            # 热力图显示轨迹密度
            all_points = []
            for i in range(num_samples):
                traj = trajs[i].numpy()
                all_points.extend(traj.tolist())
            
            all_points = np.array(all_points)
            heatmap, xedges, yedges = np.histogram2d(
                all_points[:, 0], all_points[:, 1], 
                bins=30, range=[[-0.7, 0.7], [-0.7, 0.7]])
            
            im = ax.imshow(heatmap.T, origin='lower', 
                          extent=[-0.7, 0.7, -0.7, 0.7],
                          cmap='YlOrRd', alpha=0.7, aspect='equal')
            plt.colorbar(im, ax=ax, label='Point Density', shrink=0.8)
            ax.set_title('Trajectory Density Heatmap', fontsize=12, fontweight='bold')
        
        # 起点终点
        ax.scatter(start[0], start[1], c='#27ae60', s=150, marker='s', 
                   zorder=10, edgecolors='white', linewidth=2)
        ax.scatter(goal[0], goal[1], c='gold', s=200, marker='*', 
                   zorder=10, edgecolors='black', linewidth=1.5)
    
    fig.suptitle('MPD Multi-Sample Trajectory Diversity Analysis', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("\n" + "=" * 60)
    print("  Diffusion Model Convergence Visualization - Enhanced")
    print("  For Paper: Dual Diffusion Dynamic Obstacle Avoidance")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    # 加载模型
    print("\n[1/6] Loading models...")
    upper = UpperDiffusionWithProcess(device).load()
    lower = LowerDiffusionWithProcess(device).load()
    
    # 生成测试数据
    print("\n[2/6] Generating test scenario...")
    config = ObstacleMotionConfig(arena_size=0.6, num_obstacles=6, 
                                   obs_history_len=8, pred_horizon=12)
    gen = ComplexObstacleDataGenerator(config)
    
    # 生成一个障碍物的历史和未来轨迹
    traj = gen.generate_single_obstacle_trajectory('circular', total_steps=50)
    obs_history = traj[:8].to(device)  # 历史8帧
    ground_truth = traj[8:20].to(device)  # 未来12帧真实值
    
    # 生成多个障碍物当前位置
    obstacles = []
    for i in range(6):
        t = gen.generate_single_obstacle_trajectory('random_walk', total_steps=20)
        obstacles.append(t[10].to(device))
    
    start = torch.tensor([-0.4, -0.4])
    goal = torch.tensor([0.4, 0.4])
    
    print(f"  History shape: {obs_history.shape}")
    print(f"  Ground truth shape: {ground_truth.shape}")
    print(f"  Obstacles: {len(obstacles)}")
    
    # 记录去噪过程
    print("\n[3/6] Recording diffusion processes...")
    
    # 上层模型是50步，下层是100步
    upper_record_steps = [49, 40, 30, 20, 10, 0]  # 上层50步
    lower_record_steps = [99, 80, 60, 40, 20, 0]  # 下层100步
    
    _, upper_snapshots = upper.predict_with_process(obs_history, num_samples=1, 
                                                     record_steps=upper_record_steps)
    _, lower_snapshots = lower.plan_with_process(start, goal, obstacles, 
                                                  num_samples=1, 
                                                  record_steps=lower_record_steps)
    
    print(f"  Upper snapshots: {len(upper_snapshots)}")
    print(f"  Lower snapshots: {len(lower_snapshots)}")
    
    # 创建输出目录
    out_dir = '/home/wujiahao/mpd-build/dynamic_mpd/results/diffusion_process'
    os.makedirs(out_dir, exist_ok=True)
    
    # 生成可视化
    print("\n[4/6] Creating main visualizations...")
    
    # 1. 上层扩散过程图（增强版）
    create_upper_diffusion_figure(
        upper_snapshots, obs_history, ground_truth,
        os.path.join(out_dir, 'upper_diffusion_process.png')
    )
    
    # 2. 下层扩散过程图（增强版）
    create_lower_diffusion_figure(
        lower_snapshots, start.numpy(), goal.numpy(), obstacles,
        os.path.join(out_dir, 'lower_diffusion_process.png')
    )
    
    # 3. 并排对比图（论文级）
    create_side_by_side_convergence(
        upper_snapshots, lower_snapshots,
        obs_history, ground_truth,
        start.numpy(), goal.numpy(), obstacles,
        os.path.join(out_dir, 'dual_diffusion_convergence.png')
    )
    
    # 4. 噪声调度可视化
    print("\n[5/6] Creating additional visualizations...")
    create_noise_schedule_figure(
        os.path.join(out_dir, 'noise_schedule.png')
    )
    
    # 5. 多样本多样性分析
    create_multi_sample_diversity_figure(
        lower, start, goal, obstacles,
        os.path.join(out_dir, 'trajectory_diversity.png')
    )
    
    # 6. 联合动画GIF
    print("\n[6/6] Creating animation...")
    create_combined_animation(
        upper_snapshots, lower_snapshots,
        obs_history, ground_truth,
        start.numpy(), goal.numpy(), obstacles,
        os.path.join(out_dir, 'dual_diffusion_animation.gif')
    )
    
    print("\n" + "=" * 60)
    print("  Visualization Complete!")
    print(f"  Output directory: {out_dir}")
    print("\n  Generated files:")
    print("    📊 upper_diffusion_process.png    - Upper model denoising")
    print("    📊 lower_diffusion_process.png    - Lower MPD denoising")
    print("    📊 dual_diffusion_convergence.png - Side-by-side comparison")
    print("    📊 noise_schedule.png             - Noise schedule curves")
    print("    📊 trajectory_diversity.png       - Multi-sample analysis")
    print("    🎬 dual_diffusion_animation.gif   - Animation")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
ROSOrin智能小车 - 激光雷达导航与双层扩散模型避障

特点：
1. 模拟激光雷达感知 - 只能感知有限范围内的障碍物
2. 未知环境探索 - 初始只知道目标点方位，不知道完整地图
3. 实时感知与规划 - 使用双层扩散模型进行动态避障
4. 动态重规划 - 发现新障碍物时及时重新规划路径

Author: Dual Diffusion Navigation System
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
from matplotlib.patches import Circle, Wedge, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from collections import deque
import time

from dotmap import DotMap
from mpd.utils.loaders import load_params_from_yaml
from torch_robotics.torch_utils.torch_utils import freeze_torch_model_params

from src.trainable_obstacle_diffusion import TrainableObstacleDiffusion, DiffusionConfig
from src.complex_obstacle_data import ComplexObstacleDataGenerator, ObstacleMotionConfig

print("[OK] Dependencies loaded!")

# 设置随机种子，确保可复现
SEED = 36
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"[OK] Random seed set to {SEED}")

# Matplotlib设置
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


def to_torch(x, device='cpu', dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class LidarSensor:
    """模拟ROSOrin小车的激光雷达传感器"""
    
    def __init__(self, max_range=0.35, num_rays=360, fov=360, device='cuda'):
        """
        Args:
            max_range: 激光雷达最大感知距离 (归一化坐标)
            num_rays: 激光射线数量
            fov: 视场角（度）
            device: 计算设备
        """
        self.max_range = max_range
        self.num_rays = num_rays
        self.fov = np.radians(fov)
        self.device = device
        
        # 预计算射线角度
        self.ray_angles = np.linspace(-self.fov/2, self.fov/2, num_rays)
        
    def scan(self, robot_pos, obstacles, obstacle_radii=None):
        """
        执行一次激光雷达扫描
        
        Args:
            robot_pos: 机器人位置 [x, y]
            obstacles: 障碍物位置列表 [(x, y), ...]
            obstacle_radii: 障碍物半径列表
            
        Returns:
            detected_obstacles: 检测到的障碍物列表 [(pos, dist), ...]
            scan_points: 扫描点云 [N, 2]
            ranges: 每条射线的距离
        """
        robot_pos = np.array(robot_pos) if not isinstance(robot_pos, np.ndarray) else robot_pos
        
        if obstacle_radii is None:
            obstacle_radii = [0.08] * len(obstacles)
        
        detected = []
        detected_ids = set()
        ranges = np.full(self.num_rays, self.max_range)
        scan_points = []
        
        for ray_idx, angle in enumerate(self.ray_angles):
            # 射线方向
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            
            # 检测每个障碍物
            min_dist = self.max_range
            hit_obs_idx = -1
            
            for obs_idx, obs in enumerate(obstacles):
                obs_pos = np.array(obs.cpu() if torch.is_tensor(obs) else obs)
                obs_radius = obstacle_radii[obs_idx]
                
                # 射线-圆碰撞检测
                to_obs = obs_pos - robot_pos
                
                # 投影长度
                proj_len = np.dot(to_obs, ray_dir)
                if proj_len < 0:
                    continue
                
                # 垂直距离
                perp_dist = np.abs(np.cross(ray_dir, to_obs))
                
                # 检查是否击中障碍物
                if perp_dist < obs_radius:
                    # 计算精确交点距离
                    hit_dist = proj_len - np.sqrt(max(0, obs_radius**2 - perp_dist**2))
                    if 0 < hit_dist < min_dist:
                        min_dist = hit_dist
                        hit_obs_idx = obs_idx
            
            ranges[ray_idx] = min_dist
            
            # 记录扫描点（射线端点）
            end_point = robot_pos + ray_dir * min_dist
            scan_points.append(end_point)
            
            # 记录检测到的障碍物
            if hit_obs_idx >= 0 and hit_obs_idx not in detected_ids:
                detected_ids.add(hit_obs_idx)
                obs_pos = obstacles[hit_obs_idx]
                if torch.is_tensor(obs_pos):
                    obs_pos = obs_pos.cpu().numpy()
                dist = np.linalg.norm(obs_pos - robot_pos)
                detected.append({
                    'id': hit_obs_idx,
                    'position': obs_pos,
                    'distance': dist,
                    'radius': obstacle_radii[hit_obs_idx]
                })
        
        scan_points = np.array(scan_points) if scan_points else np.zeros((0, 2))
        
        # 返回射线角度用于可视化
        return detected, scan_points, ranges, self.ray_angles.copy()


class LocalMap:
    """局部地图 - 存储已发现的障碍物及其历史轨迹"""
    
    def __init__(self, history_len=8, device='cuda'):
        self.history_len = history_len
        self.device = device
        
        # 已发现的障碍物 {id: {'history': deque, 'last_seen': step, 'radius': r}}
        self.obstacles = {}
        
        # 静态障碍物（墙壁等）
        self.static_obstacles = []
        
    def update(self, detected_obstacles, current_step):
        """更新局部地图"""
        for det in detected_obstacles:
            obs_id = det['id']
            pos = det['position']
            
            if obs_id not in self.obstacles:
                # 新发现的障碍物
                self.obstacles[obs_id] = {
                    'history': deque(maxlen=self.history_len),
                    'last_seen': current_step,
                    'radius': det['radius'],
                    'is_dynamic': False,  # 初始假设是静态的
                    'first_pos': pos.copy()
                }
            
            # 更新历史
            self.obstacles[obs_id]['history'].append(pos.copy())
            self.obstacles[obs_id]['last_seen'] = current_step
            
            # 检测是否为动态障碍物
            if len(self.obstacles[obs_id]['history']) >= 3:
                positions = list(self.obstacles[obs_id]['history'])
                movement = np.linalg.norm(np.array(positions[-1]) - np.array(positions[0]))
                if movement > 0.02:  # 移动超过阈值则判定为动态
                    self.obstacles[obs_id]['is_dynamic'] = True
    
    def get_visible_obstacles(self, current_step, max_age=10):
        """获取最近可见的障碍物"""
        visible = []
        for obs_id, obs_data in self.obstacles.items():
            if current_step - obs_data['last_seen'] <= max_age:
                if len(obs_data['history']) > 0:
                    visible.append({
                        'id': obs_id,
                        'position': obs_data['history'][-1],
                        'history': list(obs_data['history']),
                        'is_dynamic': obs_data['is_dynamic'],
                        'radius': obs_data['radius']
                    })
        return visible
    
    def get_obstacle_history(self, obs_id):
        """获取特定障碍物的历史轨迹"""
        if obs_id in self.obstacles:
            return list(self.obstacles[obs_id]['history'])
        return []
    
    def get_dynamic_obstacles(self):
        """获取所有动态障碍物"""
        return [obs_id for obs_id, data in self.obstacles.items() if data['is_dynamic']]


class DualDiffusionNavigator:
    """双层扩散模型导航器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tensor_args = {'device': device, 'dtype': torch.float32}
        
        # 加载模型
        self.upper_model = None
        self.lower_model = None
        self.upper_config = None
        
        # 规划参数
        self.replan_interval = 5
        self.cached_trajectory = None
        self.trajectory_progress = 0
        self.last_replan_step = -100
        
        # 新障碍物触发重规划
        self.known_obstacle_ids = set()
        
    def load_models(self):
        """加载双层扩散模型"""
        print("  Loading dual diffusion models...")
        
        # 上层：障碍物预测模型
        upper_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'old_results/trained_diffusion_model.pth')
        ckpt = torch.load(upper_path, map_location=self.device)
        self.upper_config = ckpt['config']
        self.upper_model = TrainableObstacleDiffusion(self.upper_config, self.device)
        self.upper_model.denoise_net.load_state_dict(ckpt['model_state_dict'])
        self.upper_model.denoise_net.eval()
        for p in self.upper_model.denoise_net.parameters():
            p.requires_grad = False
        print(f"    [Upper] Loaded: epochs={ckpt['num_epochs']}, loss={ckpt['best_test_loss']:.4f}")
        
        # 下层：轨迹规划模型
        cfg_path = os.path.join(MPD_ROOT, 'scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml')
        args_inference = DotMap(load_params_from_yaml(cfg_path))
        model_dir = os.path.expandvars(args_inference.model_dir_ddpm_bspline)
        model_path = os.path.join(model_dir, 'checkpoints', 'ema_model_current.pth')
        
        self.lower_model = torch.load(model_path, map_location=self.device)
        self.lower_model.to(self.device)
        self.lower_model.eval()
        freeze_torch_model_params(self.lower_model)
        
        if hasattr(self.lower_model, 'model') and hasattr(self.lower_model.model, 'module'):
            self.lower_model.model = self.lower_model.model.module
        
        self.n_support_points = self.lower_model.model.n_support_points
        self.state_dim = self.lower_model.state_dim
        print(f"    [Lower] Loaded: diffusion_steps={self.lower_model.n_diffusion_steps}, H={self.n_support_points}")
        
    def predict_obstacle_motion(self, obstacle_history, num_samples=12):
        """使用上层模型预测障碍物运动"""
        if len(obstacle_history) < 4:
            # 历史太短，假设静止
            last_pos = obstacle_history[-1] if len(obstacle_history) > 0 else [0, 0]
            return np.tile(last_pos, (12, 1))
        
        # 填充到所需长度
        history = np.array(obstacle_history)
        
        # 检查数据有效性
        if np.any(np.isnan(history)) or np.any(np.isinf(history)):
            return np.tile(history[-1], (12, 1))
        
        if len(history) < self.upper_config.obs_history_len:
            pad = np.tile(history[0], (self.upper_config.obs_history_len - len(history), 1))
            history = np.vstack([pad, history])
        else:
            history = history[-self.upper_config.obs_history_len:]
        
        history_tensor = torch.tensor(history, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            pred_mean, pred_std, _ = self.upper_model.predict(history_tensor, num_samples)
        
        result = pred_mean.cpu().numpy()
        
        # 检查输出有效性
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return np.tile(history[-1], (12, 1))
        
        return result
    
    def plan_trajectory(self, start, goal, obstacles, num_samples=30, record_diffusion=False, near_goal=False):
        """使用下层模型规划轨迹
        
        Args:
            record_diffusion: 如果为True，记录扩散过程中的中间状态用于可视化
            near_goal: 是否接近终点，用于调整斥力场强度
        """
        start = to_torch(start, **self.tensor_args)
        goal = to_torch(goal, **self.tensor_args)
        
        H = self.n_support_points
        
        # 准备障碍物
        if obstacles and len(obstacles) > 0:
            obs_list = []
            for obs in obstacles:
                if isinstance(obs, dict):
                    pos = obs['position']
                else:
                    pos = obs
                obs_list.append(to_torch(pos, **self.tensor_args))
            obs_tensor = torch.stack(obs_list)
            obs_normalized = obs_tensor / 0.6
        else:
            obs_normalized = None
        
        start_n = start / 0.6
        goal_n = goal / 0.6
        
        # 用于记录扩散过程
        diffusion_history = [] if record_diffusion else None
        
        with torch.no_grad():
            qs_normalized = torch.cat([start_n, goal_n]).unsqueeze(0).expand(num_samples, -1)
            context = self.lower_model.context_model(qs_normalized=qs_normalized)
            
            hard_conds = {
                0: start_n.unsqueeze(0).expand(num_samples, -1),
                H - 1: goal_n.unsqueeze(0).expand(num_samples, -1)
            }
            
            x = torch.randn(num_samples, H, self.state_dim, device=self.device)
            for idx, val in hard_conds.items():
                x[:, idx, :] = val
            
            # 终点附近减弱斥力场引导
            guidance_scale = 0.1 if near_goal else 0.3
            obs_radius_n = 0.12 / 0.6
            
            # 记录初始噪声状态
            if record_diffusion:
                diffusion_history.append({
                    'step': self.lower_model.n_diffusion_steps,
                    'x': (x * 0.6).cpu().numpy().copy(),
                    'x0_pred': None
                })
            
            for t in reversed(range(self.lower_model.n_diffusion_steps)):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                eps_pred = self.lower_model.model(x, t_batch, context=context)
                
                alpha = self.lower_model.alphas_cumprod[t]
                alpha_prev = self.lower_model.alphas_cumprod_prev[t]
                beta = self.lower_model.betas[t]
                
                x0_pred = (x - torch.sqrt(1 - alpha) * eps_pred) / torch.sqrt(alpha)
                x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
                
                # 障碍物引导 - 终点附近减弱
                activation_threshold = 30 if near_goal else 60  # 终点附近更晚激活
                if obs_normalized is not None and t < activation_threshold:
                    guidance = torch.zeros_like(x0_pred)
                    safety_mult = 1.5 if near_goal else 2.5  # 终点附近缩小安全距离
                    for obs_pos in obs_normalized:
                        diff = x0_pred - obs_pos.view(1, 1, 2)
                        dist = torch.norm(diff, dim=-1, keepdim=True)
                        safety = obs_radius_n * safety_mult
                        mask = (dist < safety).float()
                        rep_dir = diff / (dist + 1e-4)
                        rep_str = mask * (safety - dist) / safety
                        guidance += rep_dir * rep_str
                    guidance = torch.clamp(guidance, -1.0, 1.0)
                    t_weight = (activation_threshold - t) / float(activation_threshold)
                    x0_pred = x0_pred + guidance_scale * t_weight * guidance
                
                mean = (torch.sqrt(alpha_prev) * beta / (1 - alpha) * x0_pred +
                        torch.sqrt(alpha) * (1 - alpha_prev) / (1 - alpha) * x)
                
                if t > 0:
                    variance = self.lower_model.posterior_variance[t]
                    x = mean + torch.sqrt(variance) * torch.randn_like(x)
                else:
                    x = mean
                
                for idx, val in hard_conds.items():
                    x[:, idx, :] = val
                
                # 记录扩散过程（在关键时刻）
                if record_diffusion and t in [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 0]:
                    diffusion_history.append({
                        'step': t,
                        'x': (x * 0.6).cpu().numpy().copy(),
                        'x0_pred': (x0_pred * 0.6).cpu().numpy().copy()
                    })
            
            trajs = x * 0.6
        
        # 选择最佳轨迹
        best = self._select_best_trajectory(trajs, goal, obstacles)
        
        if record_diffusion:
            return best.cpu().numpy(), diffusion_history
        return best.cpu().numpy()
    
    def _select_best_trajectory(self, trajs, goal, obstacles, obs_radius=0.10):
        """选择最佳轨迹 - 优化版：终点附近自适应评分"""
        scores = []
        
        # 计算起点到终点的距离，用于判断是否接近终点
        start_pos = trajs[0][0] if len(trajs) > 0 and len(trajs[0]) > 0 else goal
        dist_to_goal_start = torch.norm(start_pos - goal).item()
        near_goal = dist_to_goal_start < 0.25  # 接近终点标志
        
        for traj in trajs:
            # 路径长度
            lengths = torch.norm(traj[1:] - traj[:-1], dim=1)
            path_len = lengths.sum().item()
            
            # 目标距离 - 接近终点时权重大幅增加
            goal_dist = torch.norm(traj[-1] - goal).item()
            goal_weight = 10.0 if near_goal else 3.0  # 终点附近大幅增加权重，优先到达
            
            # 碰撞惩罚 - 接近终点时大幅降低安全距离
            collision = 0
            safety_factor = 1.1 if near_goal else 1.8  # 终点附近仅避免真正碰撞
            
            if obstacles:
                for obs in obstacles:
                    if isinstance(obs, dict):
                        obs_pos = to_torch(obs['position'], **self.tensor_args)
                        r = obs.get('radius', obs_radius)
                    else:
                        obs_pos = to_torch(obs, **self.tensor_args)
                        r = obs_radius
                    
                    dists = torch.norm(traj - obs_pos, dim=1)
                    collision += torch.sum(torch.relu(r * safety_factor - dists) ** 2).item()
                    if dists.min().item() < r:
                        collision += 30.0 if near_goal else 50.0  # 终点附近降低惩罚
            
            # 平滑度
            if len(traj) > 2:
                v1 = traj[1:-1] - traj[:-2]
                v2 = traj[2:] - traj[1:-1]
                turns = torch.sum(torch.abs(v1 - v2)).item()
            else:
                turns = 0
            
            # 接近终点时，大幅优先考虑到达终点，基本忽略碰撞惩罚（仅避免真正碰撞）
            collision_weight = 2.0 if near_goal else 8.0
            path_weight = 0.1 if near_goal else 0.3  # 终点附近不在乎路径长度
            score = goal_dist * goal_weight + path_len * path_weight + collision * collision_weight + turns * 0.1
            scores.append(score)
        
        return trajs[np.argmin(scores)]
    
    def should_replan(self, current_step, visible_obstacles):
        """判断是否需要重新规划"""
        # 定期重规划
        if current_step - self.last_replan_step >= self.replan_interval:
            return True, "periodic"
        
        # 轨迹用完
        if self.cached_trajectory is None or self.trajectory_progress >= len(self.cached_trajectory) - 2:
            return True, "trajectory_exhausted"
        
        # 发现新障碍物
        current_ids = set(obs['id'] for obs in visible_obstacles)
        new_obstacles = current_ids - self.known_obstacle_ids
        if new_obstacles:
            self.known_obstacle_ids = current_ids
            return True, f"new_obstacles: {new_obstacles}"
        
        return False, None
    
    def get_control(self, robot_pos, goal, visible_obstacles, current_step):
        """获取控制命令"""
        robot_pos = np.array(robot_pos)
        goal = np.array(goal)
        
        dist_to_goal = np.linalg.norm(robot_pos - goal)
        
        # 检查是否需要重规划
        need_replan, reason = self.should_replan(current_step, visible_obstacles)
        
        if need_replan:
            # 收集所有障碍物（当前位置 + 预测位置）
            all_obstacles = []
            
            for obs in visible_obstacles:
                # 当前位置
                all_obstacles.append(obs)
                
                # 如果是动态障碍物，添加预测位置
                if obs['is_dynamic'] and len(obs['history']) >= 4:
                    predictions = self.predict_obstacle_motion(obs['history'])
                    for t_idx in [3, 6, 9, 11]:
                        if t_idx < len(predictions):
                            all_obstacles.append({
                                'position': predictions[t_idx],
                                'radius': obs['radius']
                            })
            
            # 判断是否接近终点
            dist_to_goal_plan = np.linalg.norm(robot_pos - goal)
            is_near_goal = dist_to_goal_plan < 0.25
            
            # 规划新轨迹 - 传入near_goal参数以调整斥力场强度
            self.cached_trajectory = self.plan_trajectory(robot_pos, goal, all_obstacles, near_goal=is_near_goal)
            self.trajectory_progress = 0
            self.last_replan_step = current_step
        
        # 紧急避障检查：如果有障碍物非常近，优先避开
        # 接近终点时降低紧急避障敏感度，优先到达目标
        dist_to_goal = np.linalg.norm(robot_pos - goal)
        near_goal = dist_to_goal < 0.2
        
        emergency_avoidance = None
        for obs in visible_obstacles:
            obs_pos = np.array(obs['position'])
            dist = np.linalg.norm(robot_pos - obs_pos)
            # 接近终点时使用更小的安全距离
            safe_dist = obs['radius'] + (0.05 if near_goal else 0.08)
            
            if dist < safe_dist:
                # 紧急避障：直接远离障碍物
                avoid_dir = robot_pos - obs_pos
                if np.linalg.norm(avoid_dir) > 1e-6:
                    avoid_dir = avoid_dir / np.linalg.norm(avoid_dir)
                    emergency_avoidance = avoid_dir * 0.03
                break
        
        if emergency_avoidance is not None:
            return emergency_avoidance, True, "emergency_avoidance"
        
        # 跟随轨迹
        if self.cached_trajectory is not None and self.trajectory_progress < len(self.cached_trajectory) - 1:
            lookahead = min(self.trajectory_progress + 2, len(self.cached_trajectory) - 1)
            target = self.cached_trajectory[lookahead]
            
            direction = target - robot_pos
            dist = np.linalg.norm(direction)
            
            if dist > 1e-6:
                direction = direction / dist
            
            # 如果接近目标，增加直接朝目标的权重
            if dist_to_goal < 0.3:
                goal_dir = goal - robot_pos
                goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
                # 混合方向 - 越接近终点，越偏向目标
                weight = max(0.4, 1.0 - dist_to_goal / 0.3)
                direction = (1 - weight) * direction + weight * goal_dir
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            # 速度控制
            if dist_to_goal < 0.15:
                speed = 0.025 if dist_to_goal > 0.05 else 0.015
            else:
                speed = 0.025
            
            self.trajectory_progress += 1
            return direction * speed, need_replan, reason
        else:
            # 直接朝目标
            direction = goal - robot_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                direction = direction / dist * 0.02
            return direction, need_replan, reason


def create_environment(n_static=3, n_dynamic=3, device='cuda'):
    """创建环境：静态和动态障碍物"""
    
    # 生成动态障碍物轨迹（远离起点和主路径）
    config = ObstacleMotionConfig(
        arena_size=0.5,  # 缩小范围
        num_obstacles=n_dynamic,
        obs_history_len=8,
        pred_horizon=12
    )
    gen = ComplexObstacleDataGenerator(config)
    
    motion_types = ['circular', 'oscillating', 'linear']
    dynamic_trajs = []
    
    # 动态障碍物的偏移，避免出现在起点附近
    offsets = [
        torch.tensor([0.15, 0.0], device=device),
        torch.tensor([-0.1, 0.2], device=device),
        torch.tensor([0.1, -0.15], device=device),
    ]
    
    for i in range(n_dynamic):
        motion = motion_types[i % len(motion_types)]
        traj = gen.generate_single_obstacle_trajectory(motion, total_steps=500)
        # 添加偏移
        traj = traj.to(device) + offsets[i % len(offsets)]
        # 限制在场地内
        traj = torch.clamp(traj, -0.55, 0.55)
        dynamic_trajs.append(traj)
    
    # 静态障碍物（避开起点区域和对角线路径）
    static_positions = [
        torch.tensor([0.15, 0.35], device=device),
        torch.tensor([0.3, -0.15], device=device),
        torch.tensor([-0.25, 0.1], device=device),
    ][:n_static]
    
    return static_positions, dynamic_trajs


def run_simulation(max_steps=400):
    """运行仿真"""
    print("\n" + "=" * 70)
    print("  ROSOrin LiDAR Navigation with Dual Diffusion Model")
    print("  - Unknown environment exploration")
    print("  - Real-time obstacle detection and prediction")
    print("  - Dynamic replanning on new obstacle discovery")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    # 初始化组件
    lidar = LidarSensor(max_range=0.35, num_rays=180, device=device)
    local_map = LocalMap(history_len=8, device=device)
    navigator = DualDiffusionNavigator(device=device)
    navigator.load_models()
    
    # 创建环境
    print("\n  Creating environment...")
    static_obs, dynamic_trajs = create_environment(n_static=4, n_dynamic=4, device=device)
    n_dynamic = len(dynamic_trajs)
    n_static = len(static_obs)
    n_total = n_static + n_dynamic
    
    # 障碍物半径
    obs_radii = [0.08] * n_total
    
    # 起点和终点
    start = np.array([-0.45, -0.45])
    goal = np.array([0.45, 0.45])
    
    print(f"  Start: {start}")
    print(f"  Goal:  {goal}")
    print(f"  Static obstacles: {n_static}")
    print(f"  Dynamic obstacles: {n_dynamic}")
    print(f"  LiDAR range: {lidar.max_range}")
    
    # 仿真状态
    robot_pos = start.copy()
    history = {'robot': [robot_pos.copy()], 
               'scans': [], 
               'trajectories': [],
               'replan_steps': [],
               'detected_obs': [],
               'ray_angles': [],
               'ranges': [],
               'explored_cells': set()}  # 已探索的网格单元
    
    print("\n  Running simulation...")
    
    for step in range(max_steps):
        # 获取当前所有障碍物位置
        all_obstacles = []
        for i, static_pos in enumerate(static_obs):
            all_obstacles.append(static_pos)
        for i, dyn_traj in enumerate(dynamic_trajs):
            all_obstacles.append(dyn_traj[min(step, len(dyn_traj)-1)])
        
        # 激光雷达扫描
        detected, scan_points, ranges, ray_angles = lidar.scan(robot_pos, all_obstacles, obs_radii)
        
        # 更新局部地图
        local_map.update(detected, step)
        
        # 获取可见障碍物
        visible_obs = local_map.get_visible_obstacles(step, max_age=15)
        
        # 获取控制命令
        control, replanned, reason = navigator.get_control(
            robot_pos, goal, visible_obs, step
        )
        
        # 更新位置
        robot_pos = robot_pos + control
        
        # 检查位置有效性
        if np.any(np.isnan(robot_pos)) or np.any(np.isinf(robot_pos)):
            print(f"\n  [Step {step}] Invalid position detected, resetting...")
            robot_pos = history['robot'][-1].copy()  # 恢复到上一个有效位置
            continue
        
        # 限制在场地内
        robot_pos = np.clip(robot_pos, -0.6, 0.6)
        
        # 记录历史
        history['robot'].append(robot_pos.copy())
        history['scans'].append(scan_points.copy())
        history['detected_obs'].append([obs['position'].copy() for obs in visible_obs])
        history['ray_angles'].append(ray_angles.copy())
        history['ranges'].append(ranges.copy())
        
        # 更新已探索区域（网格化）
        grid_resolution = 0.05
        for sp in scan_points:
            # 从机器人到扫描点的所有网格都标记为已探索
            line_points = np.linspace(robot_pos, sp, 20)
            for pt in line_points:
                cell = (int((pt[0] + 0.7) / grid_resolution), int((pt[1] + 0.7) / grid_resolution))
                history['explored_cells'].add(cell)
        
        if replanned:
            history['replan_steps'].append(step)
            history['trajectories'].append(navigator.cached_trajectory.copy() if navigator.cached_trajectory is not None else None)
        
        # 检查是否到达目标
        dist_to_goal = np.linalg.norm(robot_pos - goal)
        if dist_to_goal < 0.05:
            print(f"\n  [Step {step}] GOAL REACHED! Final distance: {dist_to_goal:.4f}")
            break
        
        # 检查碰撞
        collision = False
        for obs in all_obstacles:
            obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
            if np.linalg.norm(robot_pos - obs_np) < 0.08:
                collision = True
                print(f"\n  [Step {step}] COLLISION DETECTED!")
                break
        
        if collision:
            break
        
        # 打印进度
        if step % 30 == 0:
            n_detected = len(visible_obs)
            n_dynamic_detected = sum(1 for o in visible_obs if o['is_dynamic'])
            print(f"  [Step {step:3d}] Dist: {dist_to_goal:.3f} | "
                  f"Detected: {n_detected} ({n_dynamic_detected} dynamic) | "
                  f"Replans: {len(history['replan_steps'])}")
    
    # 统计结果
    total_steps = len(history['robot']) - 1
    final_dist = np.linalg.norm(history['robot'][-1] - goal)
    total_replans = len(history['replan_steps'])
    
    print(f"\n  Simulation complete:")
    print(f"    Total steps: {total_steps}")
    print(f"    Final distance: {final_dist:.4f}")
    print(f"    Total replans: {total_replans}")
    print(f"    Success: {'Yes' if final_dist < 0.1 else 'No'}")
    
    return history, static_obs, dynamic_trajs, lidar, start, goal


def draw_fog_of_war(ax, explored_cells, grid_resolution=0.05, arena_min=-0.7, arena_max=0.7):
    """绘制迷雾效果：未探索区域用半透明深灰色覆盖"""
    grid_size_x = int((arena_max - arena_min) / grid_resolution)
    grid_size_y = int((arena_max - arena_min) / grid_resolution)
    
    # 创建迷雾网格（1=未探索，0=已探索）
    fog = np.ones((grid_size_y, grid_size_x))
    
    for cell in explored_cells:
        cx, cy = cell
        if 0 <= cx < grid_size_x and 0 <= cy < grid_size_y:
            fog[cy, cx] = 0
    
    # 平滑迷雾边缘
    from scipy.ndimage import gaussian_filter
    fog_smooth = gaussian_filter(fog, sigma=1.5)
    
    # 绘制迷雾
    extent = [arena_min, arena_max, arena_min, arena_max]
    ax.imshow(fog_smooth, extent=extent, origin='lower', cmap='Greys', 
              alpha=0.7, vmin=0, vmax=1, zorder=100)


def draw_lidar_rays(ax, robot_pos, ray_angles, ranges, max_range, color='cyan', alpha=0.3):
    """绘制激光雷达射线"""
    for i, (angle, r) in enumerate(zip(ray_angles, ranges)):
        # 射线方向
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        end_point = robot_pos + ray_dir * r
        
        # 根据是否碰到障碍物选择颜色
        if r < max_range - 0.01:
            # 碰到障碍物：红色
            ax.plot([robot_pos[0], end_point[0]], [robot_pos[1], end_point[1]], 
                   color='red', alpha=alpha*0.8, linewidth=0.5, zorder=50)
        else:
            # 未碰到：青色
            ax.plot([robot_pos[0], end_point[0]], [robot_pos[1], end_point[1]], 
                   color=color, alpha=alpha*0.5, linewidth=0.3, zorder=50)


def create_visualization(history, static_obs, dynamic_trajs, lidar, start, goal):
    """创建可视化"""
    print("\n  Creating visualization...")
    
    out_dir = '/home/wujiahao/mpd-build/dynamic_mpd/results/lidar_navigation'
    os.makedirs(out_dir, exist_ok=True)
    
    n_static = len(static_obs)
    n_dynamic = len(dynamic_trajs)
    total_steps = len(history['robot']) - 1
    
    obs_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#16a085']
    robot_traj = np.array(history['robot'])
    
    # ========== 图1: 完整轨迹图（带迷雾和激光线） ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 左图：最终探索地图
    ax = axes[0]
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.set_facecolor('#2c3e50')  # 深色背景代表未知区域
    ax.grid(True, alpha=0.2, color='white')
    ax.set_title('Final Explored Map', fontweight='bold')
    
    # 绘制已探索区域（浅色）
    explored_cells = history['explored_cells']
    if explored_cells:
        # 绘制已探索的地面
        grid_resolution = 0.05
        arena_min, arena_max = -0.7, 0.7
        grid_size = int((arena_max - arena_min) / grid_resolution)
        explored_map = np.zeros((grid_size, grid_size))
        
        for cell in explored_cells:
            cx, cy = cell
            if 0 <= cx < grid_size and 0 <= cy < grid_size:
                explored_map[cy, cx] = 1
        
        # 平滑处理
        from scipy.ndimage import gaussian_filter
        explored_smooth = gaussian_filter(explored_map, sigma=1.0)
        extent = [arena_min, arena_max, arena_min, arena_max]
        ax.imshow(explored_smooth, extent=extent, origin='lower', 
                 cmap='Greens', alpha=0.4, vmin=0, vmax=1, zorder=1)
    
    # 静态障碍物（只有在已探索区域才完全显示）
    for i, obs in enumerate(static_obs):
        pos = obs.cpu().numpy()
        cell = (int((pos[0] + 0.7) / 0.05), int((pos[1] + 0.7) / 0.05))
        is_explored = cell in explored_cells
        alpha_val = 0.8 if is_explored else 0.15
        c = Circle(pos, 0.08, color='#7f8c8d', alpha=alpha_val, zorder=5)
        ax.add_patch(c)
        if is_explored:
            ax.text(pos[0], pos[1]+0.1, f'S{i+1}', ha='center', fontsize=8, color='white')
    
    # 动态障碍物最终位置
    for i, traj in enumerate(dynamic_trajs):
        final_pos = traj[min(total_steps, len(traj)-1)].cpu().numpy()
        cell = (int((final_pos[0] + 0.7) / 0.05), int((final_pos[1] + 0.7) / 0.05))
        is_explored = cell in explored_cells
        alpha_val = 0.8 if is_explored else 0.15
        c = Circle(final_pos, 0.08, color=obs_colors[i], alpha=alpha_val, zorder=5)
        ax.add_patch(c)
    
    # 机器人轨迹
    ax.plot(robot_traj[:, 0], robot_traj[:, 1], 'yellow', linewidth=2.5, label='Robot Path', zorder=10)
    
    # 重规划点
    for replan_step in history['replan_steps']:
        if replan_step < len(robot_traj):
            ax.scatter(robot_traj[replan_step, 0], robot_traj[replan_step, 1], 
                      c='orange', s=30, marker='x', zorder=15)
    
    ax.scatter(*start, c='lime', s=150, marker='s', zorder=20, label='Start', edgecolors='white')
    ax.scatter(*goal, c='gold', s=200, marker='*', zorder=20, label='Goal', edgecolors='white')
    ax.legend(loc='lower left', fontsize=9, facecolor='#34495e', labelcolor='white')
    
    # 中图：LiDAR感知示意（带激光线和迷雾）
    ax2 = axes[1]
    ax2.set_xlim(-0.7, 0.7)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_aspect('equal')
    ax2.set_facecolor('#2c3e50')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.set_title('LiDAR Scan Visualization', fontweight='bold')
    
    # 选择一个中间时刻
    mid_step = min(50, total_steps-1)
    mid_pos = robot_traj[mid_step]
    
    # 计算到mid_step为止的已探索区域
    explored_at_mid = set()
    grid_resolution = 0.05
    for s in range(mid_step + 1):
        if s < len(history['scans']):
            rpos = robot_traj[s]
            for sp in history['scans'][s]:
                line_points = np.linspace(rpos, sp, 15)
                for pt in line_points:
                    cell = (int((pt[0] + 0.7) / grid_resolution), int((pt[1] + 0.7) / grid_resolution))
                    explored_at_mid.add(cell)
    
    # 绘制已探索区域
    if explored_at_mid:
        arena_min, arena_max = -0.7, 0.7
        grid_size = int((arena_max - arena_min) / grid_resolution)
        explored_map = np.zeros((grid_size, grid_size))
        for cell in explored_at_mid:
            cx, cy = cell
            if 0 <= cx < grid_size and 0 <= cy < grid_size:
                explored_map[cy, cx] = 1
        from scipy.ndimage import gaussian_filter
        explored_smooth = gaussian_filter(explored_map, sigma=1.0)
        extent = [arena_min, arena_max, arena_min, arena_max]
        ax2.imshow(explored_smooth, extent=extent, origin='lower', 
                  cmap='Greens', alpha=0.3, vmin=0, vmax=1, zorder=1)
    
    # 绘制激光雷达射线
    if mid_step < len(history['ray_angles']) and mid_step < len(history['ranges']):
        draw_lidar_rays(ax2, mid_pos, history['ray_angles'][mid_step], 
                       history['ranges'][mid_step], lidar.max_range, 
                       color='cyan', alpha=0.6)
    
    # LiDAR范围圆
    lidar_circle = Circle(mid_pos, lidar.max_range, fill=False, 
                         color='cyan', linestyle='--', linewidth=2, alpha=0.8, zorder=60)
    ax2.add_patch(lidar_circle)
    
    # 障碍物
    for i, obs in enumerate(static_obs):
        pos = obs.cpu().numpy()
        dist = np.linalg.norm(pos - mid_pos)
        cell = (int((pos[0] + 0.7) / 0.05), int((pos[1] + 0.7) / 0.05))
        is_explored = cell in explored_at_mid
        
        if dist < lidar.max_range:
            c = Circle(pos, 0.08, color='#e74c3c', alpha=0.9, zorder=55)
            ax2.add_patch(c)
            ax2.text(pos[0], pos[1], '!', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white', zorder=56)
        elif is_explored:
            c = Circle(pos, 0.08, color='#7f8c8d', alpha=0.5, zorder=5)
            ax2.add_patch(c)
        else:
            c = Circle(pos, 0.08, color='#7f8c8d', alpha=0.1, zorder=5)
            ax2.add_patch(c)
    
    for i, traj in enumerate(dynamic_trajs):
        pos = traj[mid_step].cpu().numpy()
        dist = np.linalg.norm(pos - mid_pos)
        cell = (int((pos[0] + 0.7) / 0.05), int((pos[1] + 0.7) / 0.05))
        is_explored = cell in explored_at_mid
        
        if dist < lidar.max_range:
            c = Circle(pos, 0.08, color=obs_colors[i], alpha=0.9, zorder=55)
            ax2.add_patch(c)
        elif is_explored:
            c = Circle(pos, 0.08, color=obs_colors[i], alpha=0.4, zorder=5)
            ax2.add_patch(c)
        else:
            c = Circle(pos, 0.08, color=obs_colors[i], alpha=0.1, zorder=5)
            ax2.add_patch(c)
    
    # 扫描点（激光击中点）
    if mid_step < len(history['scans']) and len(history['scans'][mid_step]) > 0:
        scans = history['scans'][mid_step]
        ax2.scatter(scans[:, 0], scans[:, 1], c='red', s=15, alpha=0.8, 
                   label='Hit Points', zorder=58, edgecolors='white', linewidth=0.5)
    
    # 机器人
    ax2.scatter(*mid_pos, c='yellow', s=150, marker='s', zorder=70, edgecolors='white', linewidth=2)
    ax2.scatter(*goal, c='gold', s=150, marker='*', zorder=60, alpha=0.7)
    ax2.legend(loc='lower left', fontsize=9, facecolor='#34495e', labelcolor='white')
    ax2.text(0.02, 0.98, f'Step: {mid_step}', transform=ax2.transAxes, va='top', 
            fontsize=10, color='white', fontweight='bold')
    
    # 右图：重规划统计
    ax3 = axes[2]
    ax3.set_facecolor('#ecf0f1')
    replan_steps = history['replan_steps']
    
    if replan_steps:
        intervals = np.diff(replan_steps)
        ax3.hist(intervals, bins=20, color='#3498db', alpha=0.7, edgecolor='white')
        ax3.set_xlabel('Replan Interval (steps)')
        ax3.set_ylabel('Count')
        ax3.set_title('Replanning Statistics', fontweight='bold')
        ax3.axvline(np.mean(intervals), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(intervals):.1f}')
        ax3.legend()
        
        # 添加统计文字
        stats_text = f"Total Replans: {len(replan_steps)}\nMin Interval: {np.min(intervals):.0f}\nMax Interval: {np.max(intervals):.0f}"
        ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, va='top', ha='right',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No replanning occurred', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Replanning Statistics', fontweight='bold')
    
    fig.suptitle('ROSOrin LiDAR Navigation with Dual Diffusion Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'navigation_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_dir}/navigation_summary.png")
    
    # ========== 图2: 动画（带迷雾和激光线） ==========
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 预计算每一帧的已探索区域
    from scipy.ndimage import gaussian_filter
    grid_resolution = 0.05
    arena_min, arena_max = -0.7, 0.7
    grid_size = int((arena_max - arena_min) / grid_resolution)
    
    explored_by_frame = [set()]  # 第0帧没有探索
    current_explored = set()
    for s in range(total_steps):
        if s < len(history['scans']):
            rpos = robot_traj[s]
            for sp in history['scans'][s]:
                line_points = np.linspace(rpos, sp, 15)
                for pt in line_points:
                    cell = (int((pt[0] + 0.7) / grid_resolution), int((pt[1] + 0.7) / grid_resolution))
                    current_explored.add(cell)
        explored_by_frame.append(current_explored.copy())
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')  # 深色背景代表未知
        
        robot_pos = robot_traj[frame]
        explored_now = explored_by_frame[min(frame, len(explored_by_frame)-1)]
        
        # 绘制已探索区域
        if explored_now:
            explored_map = np.zeros((grid_size, grid_size))
            for cell in explored_now:
                cx, cy = cell
                if 0 <= cx < grid_size and 0 <= cy < grid_size:
                    explored_map[cy, cx] = 1
            explored_smooth = gaussian_filter(explored_map, sigma=1.0)
            extent = [arena_min, arena_max, arena_min, arena_max]
            ax.imshow(explored_smooth, extent=extent, origin='lower', 
                     cmap='Greens', alpha=0.4, vmin=0, vmax=1, zorder=1)
        
        # 绘制迷雾（未探索区域）
        fog_map = np.ones((grid_size, grid_size))
        for cell in explored_now:
            cx, cy = cell
            if 0 <= cx < grid_size and 0 <= cy < grid_size:
                fog_map[cy, cx] = 0
        fog_smooth = gaussian_filter(fog_map, sigma=1.5)
        ax.imshow(fog_smooth, extent=[arena_min, arena_max, arena_min, arena_max], 
                 origin='lower', cmap='Greys', alpha=0.6, vmin=0, vmax=1, zorder=2)
        
        # 绘制激光雷达射线
        if frame < len(history['ray_angles']) and frame < len(history['ranges']):
            draw_lidar_rays(ax, robot_pos, history['ray_angles'][frame], 
                           history['ranges'][frame], lidar.max_range, 
                           color='cyan', alpha=0.5)
        
        # LiDAR范围圆
        lidar_circle = Circle(robot_pos, lidar.max_range, fill=False, 
                             color='cyan', linestyle='--', linewidth=2, alpha=0.7, zorder=60)
        ax.add_patch(lidar_circle)
        
        # 静态障碍物
        for i, obs in enumerate(static_obs):
            pos = obs.cpu().numpy()
            dist = np.linalg.norm(pos - robot_pos)
            cell = (int((pos[0] + 0.7) / 0.05), int((pos[1] + 0.7) / 0.05))
            is_explored = cell in explored_now
            
            if dist < lidar.max_range:
                # 当前正在检测到：高亮红色
                c = Circle(pos, 0.08, color='#e74c3c', alpha=0.9, zorder=55)
                ax.add_patch(c)
            elif is_explored:
                # 之前探索过：灰色
                c = Circle(pos, 0.08, color='#7f8c8d', alpha=0.6, zorder=10)
                ax.add_patch(c)
            # 未探索的不显示（被迷雾覆盖）
        
        # 动态障碍物
        for i, traj in enumerate(dynamic_trajs):
            pos = traj[min(frame, len(traj)-1)].cpu().numpy()
            dist = np.linalg.norm(pos - robot_pos)
            cell = (int((pos[0] + 0.7) / 0.05), int((pos[1] + 0.7) / 0.05))
            is_explored = cell in explored_now
            
            if dist < lidar.max_range:
                c = Circle(pos, 0.08, color=obs_colors[i], alpha=0.9, zorder=55)
                ax.add_patch(c)
            elif is_explored:
                c = Circle(pos, 0.08, color=obs_colors[i], alpha=0.4, zorder=10)
                ax.add_patch(c)
        
        # 扫描点（激光击中点）
        if frame < len(history['scans']) and len(history['scans'][frame]) > 0:
            scans = history['scans'][frame]
            ax.scatter(scans[:, 0], scans[:, 1], c='red', s=8, alpha=0.8, zorder=58)
        
        # 机器人历史轨迹
        if frame > 0:
            ax.plot(robot_traj[:frame+1, 0], robot_traj[:frame+1, 1], 
                   'yellow', linewidth=2.5, alpha=0.8, zorder=65)
        
        # 当前规划轨迹
        for i, (step, traj) in enumerate(zip(history['replan_steps'], history['trajectories'])):
            if step <= frame and traj is not None:
                if i == len(history['replan_steps']) - 1 or history['replan_steps'][i+1] > frame:
                    ax.plot(traj[:, 0], traj[:, 1], '--', color='orange', 
                           linewidth=2, alpha=0.6, zorder=63)
                    break
        
        # 机器人
        ax.scatter(robot_pos[0], robot_pos[1], c='yellow', s=180, 
                  marker='s', zorder=70, edgecolors='white', linewidth=2)
        
        # 起点终点
        ax.scatter(*start, c='lime', s=120, marker='s', zorder=68, alpha=0.8, edgecolors='white')
        ax.scatter(*goal, c='gold', s=180, marker='*', zorder=68, edgecolors='white')
        
        # 信息
        dist_to_goal = np.linalg.norm(robot_pos - goal)
        n_detected = len(history['detected_obs'][frame]) if frame < len(history['detected_obs']) else 0
        explored_percent = len(explored_now) * 100.0 / (grid_size * grid_size)
        
        ax.set_title(f'ROSOrin LiDAR Navigation | Step {frame}/{total_steps}\n'
                    f'Distance: {dist_to_goal:.3f} | Detected: {n_detected} | Explored: {explored_percent:.1f}%',
                    fontweight='bold', fontsize=12, color='white')
        ax.set_facecolor('#1a1a2e')
        
        return []
    
    # 采样帧以加快动画生成
    frame_skip = max(1, total_steps // 100)
    frames = list(range(0, total_steps, frame_skip))
    
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    anim.save(os.path.join(out_dir, 'navigation_animation.gif'), writer='pillow', fps=10, dpi=100)
    plt.close()
    print(f"  Saved: {out_dir}/navigation_animation.gif")
    
    print(f"\n  All visualizations saved to: {out_dir}")


def visualize_diffusion_convergence(navigator, start, goal, obstacles, out_dir):
    """可视化下层扩散模型在不同时刻的轨迹收敛过程"""
    print("\n  Generating diffusion convergence visualization...")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 执行带记录的轨迹规划
    result = navigator.plan_trajectory(start, goal, obstacles, num_samples=30, record_diffusion=True)
    best_traj, diffusion_history = result
    
    # 准备障碍物位置
    obs_positions = []
    for obs in obstacles:
        if isinstance(obs, dict):
            obs_positions.append(np.array(obs['position']))
        else:
            obs_positions.append(np.array(obs.cpu().numpy() if hasattr(obs, 'cpu') else obs))
    
    # ========== 图1: 扩散过程网格图 ==========
    n_steps = len(diffusion_history)
    cols = min(4, n_steps)
    rows = (n_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # 颜色映射：从红色（噪声）到蓝色（收敛）
    cmap = plt.cm.coolwarm_r
    
    for idx, record in enumerate(diffusion_history):
        ax = axes[idx]
        step = record['step']
        x = record['x']  # shape: (num_samples, H, 2)
        
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        # 绘制障碍物
        for obs_pos in obs_positions:
            c = Circle(obs_pos, 0.08, color='#e74c3c', alpha=0.6, zorder=5)
            ax.add_patch(c)
        
        # 绘制所有采样轨迹
        num_samples = x.shape[0]
        for i in range(num_samples):
            traj = x[i]
            # 根据扩散进度选择颜色
            progress = 1.0 - step / 100.0  # 0->1 随着step减小
            color = cmap(progress)
            alpha = 0.15 + 0.35 * progress  # 越收敛越不透明
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=0.8)
        
        # 绘制轨迹均值
        mean_traj = np.mean(x, axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'k-', linewidth=2.5, alpha=0.9, 
               label='Mean Trajectory')
        
        # 起点终点
        ax.scatter(*start, c='green', s=100, marker='s', zorder=10, edgecolors='white')
        ax.scatter(*goal, c='gold', s=120, marker='*', zorder=10, edgecolors='white')
        
        # 计算轨迹方差（收敛程度指标）
        variance = np.mean(np.var(x, axis=0))
        
        ax.set_title(f't = {step}\nVariance: {variance:.4f}', fontsize=11, fontweight='bold')
        
        if idx == 0:
            ax.text(0.02, 0.98, 'Noise', transform=ax.transAxes, va='top', 
                   fontsize=9, fontweight='bold', color='red')
        elif step == 0:
            ax.text(0.02, 0.98, 'Converged', transform=ax.transAxes, va='top', 
                   fontsize=9, fontweight='bold', color='blue')
    
    # 隐藏多余的子图
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Lower Diffusion Model: Trajectory Convergence Process\n'
                 '(From Random Noise to Optimal Trajectory)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'diffusion_convergence_grid.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_dir}/diffusion_convergence_grid.png")
    
    # ========== 图2: 收敛动画 ==========
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update_diffusion(frame_idx):
        ax.clear()
        record = diffusion_history[frame_idx]
        step = record['step']
        x = record['x']
        
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#1a1a2e')
        
        # 障碍物
        for obs_pos in obs_positions:
            c = Circle(obs_pos, 0.08, color='#e74c3c', alpha=0.7, zorder=5)
            ax.add_patch(c)
        
        # 所有采样轨迹
        num_samples = x.shape[0]
        progress = 1.0 - step / 100.0
        for i in range(num_samples):
            traj = x[i]
            color = cmap(progress)
            alpha = 0.2 + 0.5 * progress
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=1)
        
        # 均值轨迹
        mean_traj = np.mean(x, axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'cyan', linewidth=3, alpha=0.9)
        
        # 起点终点
        ax.scatter(*start, c='lime', s=150, marker='s', zorder=10, edgecolors='white', linewidth=2)
        ax.scatter(*goal, c='gold', s=180, marker='*', zorder=10, edgecolors='white', linewidth=2)
        
        variance = np.mean(np.var(x, axis=0))
        ax.set_title(f'Diffusion Step: t = {step}  |  Variance: {variance:.4f}\n'
                    f'Samples: {num_samples}', fontsize=13, fontweight='bold', color='white')
        
        return []
    
    anim = animation.FuncAnimation(fig, update_diffusion, frames=len(diffusion_history), 
                                   interval=300, blit=False)
    anim.save(os.path.join(out_dir, 'diffusion_convergence_animation.gif'), 
              writer='pillow', fps=3, dpi=100)
    plt.close()
    print(f"  Saved: {out_dir}/diffusion_convergence_animation.gif")
    
    # ========== 图3: 收敛曲线分析 ==========
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    steps = [r['step'] for r in diffusion_history]
    variances = [np.mean(np.var(r['x'], axis=0)) for r in diffusion_history]
    
    # 方差随时间变化
    ax = axes[0]
    ax.plot(steps[::-1], variances[::-1], 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Diffusion Step (t → 0)', fontsize=11)
    ax.set_ylabel('Mean Trajectory Variance', fontsize=11)
    ax.set_title('Convergence Curve', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(max(steps), 0)
    
    # 轨迹多样性演变
    ax = axes[1]
    for i, record in enumerate(diffusion_history):
        x = record['x']
        step = record['step']
        # 计算轨迹间的平均距离
        mean_traj = np.mean(x, axis=0)
        distances = [np.mean(np.linalg.norm(x[j] - mean_traj, axis=1)) for j in range(x.shape[0])]
        
        color = cmap(1.0 - step / 100.0)
        ax.scatter([step] * len(distances), distances, c=[color], alpha=0.4, s=20)
    
    ax.set_xlabel('Diffusion Step', fontsize=11)
    ax.set_ylabel('Distance from Mean Trajectory', fontsize=11)
    ax.set_title('Trajectory Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(max(steps), 0)
    
    # 最终轨迹质量对比
    ax = axes[2]
    # 初始噪声
    initial = diffusion_history[0]['x']
    final = diffusion_history[-1]['x']
    
    for i in range(min(5, initial.shape[0])):
        ax.plot(initial[i, :, 0], initial[i, :, 1], 'r--', alpha=0.3, linewidth=1)
        ax.plot(final[i, :, 0], final[i, :, 1], 'b-', alpha=0.5, linewidth=1.5)
    
    # 最佳轨迹
    ax.plot(best_traj[:, 0], best_traj[:, 1], 'g-', linewidth=3, alpha=0.9, label='Best')
    
    for obs_pos in obs_positions:
        c = Circle(obs_pos, 0.08, color='#e74c3c', alpha=0.5)
        ax.add_patch(c)
    
    ax.scatter(*start, c='green', s=80, marker='s', zorder=10)
    ax.scatter(*goal, c='gold', s=100, marker='*', zorder=10)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.set_title('Initial (Red) vs Final (Blue) Trajectories', fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Lower Diffusion Model: Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'diffusion_convergence_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_dir}/diffusion_convergence_analysis.png")
    
    # ========== 图4: 不同导航时刻的扩散收敛对比 ==========
    return best_traj, diffusion_history


def visualize_multi_moment_diffusion(navigator, history, static_obs, dynamic_trajs, out_dir):
    """可视化导航过程中不同时刻的扩散模型轨迹收敛"""
    print("\n  Generating multi-moment diffusion visualization...")
    
    robot_traj = np.array(history['robot'])
    total_steps = len(robot_traj) - 1
    goal = np.array([0.45, 0.45])
    
    # 选择几个关键时刻
    key_moments = [0, total_steps // 4, total_steps // 2, 3 * total_steps // 4]
    key_moments = [m for m in key_moments if m < total_steps]
    
    fig, axes = plt.subplots(2, len(key_moments), figsize=(5*len(key_moments), 10))
    
    obs_colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12']
    cmap = plt.cm.coolwarm_r
    
    for col, moment in enumerate(key_moments):
        robot_pos = robot_traj[moment]
        
        # 获取当前时刻的障碍物
        current_obstacles = []
        for i, obs in enumerate(static_obs):
            pos = obs.cpu().numpy()
            current_obstacles.append({'position': pos, 'radius': 0.08})
        for i, traj in enumerate(dynamic_trajs):
            pos = traj[min(moment, len(traj)-1)].cpu().numpy()
            current_obstacles.append({'position': pos, 'radius': 0.08})
        
        # 执行带记录的轨迹规划
        result = navigator.plan_trajectory(robot_pos, goal, current_obstacles, 
                                          num_samples=20, record_diffusion=True)
        _, diff_history = result
        
        # 上排：初始噪声状态
        ax_top = axes[0, col]
        initial = diff_history[0]
        x_init = initial['x']
        
        ax_top.set_xlim(-0.7, 0.7)
        ax_top.set_ylim(-0.7, 0.7)
        ax_top.set_aspect('equal')
        ax_top.set_facecolor('#fff5f5')
        ax_top.grid(True, alpha=0.3)
        
        # 障碍物
        for i, obs in enumerate(current_obstacles):
            pos = obs['position']
            c = Circle(pos, 0.08, color=obs_colors[i % len(obs_colors)], alpha=0.6)
            ax_top.add_patch(c)
        
        # 初始轨迹
        for i in range(x_init.shape[0]):
            ax_top.plot(x_init[i, :, 0], x_init[i, :, 1], 'r-', alpha=0.2, linewidth=0.8)
        
        ax_top.scatter(*robot_pos, c='purple', s=100, marker='s', zorder=10, edgecolors='white')
        ax_top.scatter(*goal, c='gold', s=120, marker='*', zorder=10)
        
        var_init = np.mean(np.var(x_init, axis=0))
        ax_top.set_title(f'Step {moment}: Initial (t=100)\nVar: {var_init:.4f}', fontweight='bold')
        
        # 下排：最终收敛状态
        ax_bot = axes[1, col]
        final = diff_history[-1]
        x_final = final['x']
        
        ax_bot.set_xlim(-0.7, 0.7)
        ax_bot.set_ylim(-0.7, 0.7)
        ax_bot.set_aspect('equal')
        ax_bot.set_facecolor('#f5fff5')
        ax_bot.grid(True, alpha=0.3)
        
        # 障碍物
        for i, obs in enumerate(current_obstacles):
            pos = obs['position']
            c = Circle(pos, 0.08, color=obs_colors[i % len(obs_colors)], alpha=0.6)
            ax_bot.add_patch(c)
        
        # 最终轨迹
        for i in range(x_final.shape[0]):
            ax_bot.plot(x_final[i, :, 0], x_final[i, :, 1], 'b-', alpha=0.3, linewidth=1)
        
        # 均值轨迹
        mean_traj = np.mean(x_final, axis=0)
        ax_bot.plot(mean_traj[:, 0], mean_traj[:, 1], 'g-', linewidth=2.5, alpha=0.9)
        
        ax_bot.scatter(*robot_pos, c='purple', s=100, marker='s', zorder=10, edgecolors='white')
        ax_bot.scatter(*goal, c='gold', s=120, marker='*', zorder=10)
        
        var_final = np.mean(np.var(x_final, axis=0))
        ax_bot.set_title(f'Step {moment}: Final (t=0)\nVar: {var_final:.4f}', fontweight='bold')
    
    fig.suptitle('Lower Diffusion Model Convergence at Different Navigation Moments\n'
                 '(Top: Initial Noise | Bottom: Converged Trajectories)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'multi_moment_diffusion.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_dir}/multi_moment_diffusion.png")


def main():
    # 运行仿真
    history, static_obs, dynamic_trajs, lidar, start, goal = run_simulation(max_steps=400)
    
    out_dir = '/home/wujiahao/mpd-build/dynamic_mpd/results/lidar_navigation'
    
    # 创建导航可视化
    create_visualization(history, static_obs, dynamic_trajs, lidar, start, goal)
    
    # 创建下层扩散模型收敛可视化
    print("\n" + "-" * 50)
    print("  Generating Lower Diffusion Model Visualizations")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    navigator = DualDiffusionNavigator(device=device)
    navigator.load_models()
    
    # 准备初始障碍物（用于单次扩散收敛展示）
    initial_obstacles = []
    for obs in static_obs:
        initial_obstacles.append({'position': obs.cpu().numpy(), 'radius': 0.08})
    for traj in dynamic_trajs:
        initial_obstacles.append({'position': traj[0].cpu().numpy(), 'radius': 0.08})
    
    # 可视化单次规划的扩散收敛过程
    visualize_diffusion_convergence(navigator, start, goal, initial_obstacles, out_dir)
    
    # 可视化导航过程中不同时刻的扩散收敛
    visualize_multi_moment_diffusion(navigator, history, static_obs, dynamic_trajs, out_dir)
    
    print("\n" + "=" * 70)
    print("  ROSOrin LiDAR Navigation Complete!")
    print("  All visualizations saved to:", out_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()

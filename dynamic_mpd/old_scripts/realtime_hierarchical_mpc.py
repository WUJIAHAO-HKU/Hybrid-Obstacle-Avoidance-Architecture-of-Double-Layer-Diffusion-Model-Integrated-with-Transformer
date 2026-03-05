"""
实时层级扩散MPC框架

设计思路（按用户描述）:
┌─────────────────────────────────────────────────────────────────────────────┐
│  上层: 障碍物预测扩散模型                                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━                                                    │
│  输入: 传感器观测的障碍物位置（带噪声/不确定性）                                  │
│  过程: 扩散采样 → 多条可能轨迹 → 收敛为预测分布                                  │
│  输出: 未来障碍物分布图 (概率密度 / 置信区间)                                    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  中层: 自适应速度控制器                                                        │
│  ━━━━━━━━━━━━━━━━━━━━━                                                      │
│  状态1: 试探模式 (慢速) - 不确定性高时，优先远离障碍物                            │
│  状态2: 跟踪模式 (中速) - 找到规律后，沿安全轨迹前进                              │
│  状态3: 冲刺模式 (快速) - 路径明确时，快速到达目标                                │
│                                                                             │
│  切换条件: 基于预测置信度 + 安全裕度 + 轨迹稳定性                                │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  下层: 轨迹规划扩散模型                                                        │
│  ━━━━━━━━━━━━━━━━━━━━━                                                      │
│  输入: 当前位置 + 目标位置 + 障碍物分布图                                        │
│  过程: 从噪声采样多条轨迹 → 障碍物分布引导 → 收敛为最优轨迹                       │
│  输出: 下一步控制指令 (速度受中层约束)                                           │
└─────────────────────────────────────────────────────────────────────────────┘

实时循环: 上述过程每个控制周期重复执行 (MPC)

Author: Dynamic MPD Project
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


# ============================================================================
# 配置
# ============================================================================

class SpeedMode(Enum):
    """速度模式"""
    PROBE = "probe"      # 试探模式 - 慢速，优先避障
    TRACK = "track"      # 跟踪模式 - 中速，沿轨迹前进
    SPRINT = "sprint"    # 冲刺模式 - 快速，直奔目标


@dataclass
class MPCConfig:
    """MPC配置"""
    # 时间参数
    control_dt: float = 0.1           # 控制周期 (秒)
    prediction_horizon: float = 3.0   # 预测时域 (秒)
    planning_horizon: int = 30        # 规划步数
    
    # 速度参数 (三档速度)
    speed_probe: float = 0.02         # 试探速度
    speed_track: float = 0.04         # 跟踪速度
    speed_sprint: float = 0.06        # 冲刺速度
    
    # 模式切换阈值
    confidence_threshold_low: float = 0.3   # 低于此进入试探
    confidence_threshold_high: float = 0.7  # 高于此进入冲刺
    safety_margin: float = 0.1              # 安全距离
    
    # 扩散参数
    num_obstacle_samples: int = 20    # 障碍物预测采样数
    num_trajectory_samples: int = 15  # 轨迹规划采样数
    diffusion_steps: int = 30         # 扩散步数


# ============================================================================
# 上层: 障碍物预测扩散模型
# ============================================================================

class ObstaclePredictionDiffusion:
    """
    上层: 障碍物预测扩散模型
    
    从带噪声的观测中预测未来障碍物分布
    """
    
    def __init__(self, config: MPCConfig, device='cpu'):
        self.config = config
        self.device = device
        
        # 扩散参数
        self.num_steps = config.diffusion_steps
        self.betas = torch.linspace(1e-4, 0.02, self.num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def predict(
        self,
        observations: torch.Tensor,      # [T_obs, num_obstacles, 2] 历史观测
        prediction_steps: int = 12       # 预测未来步数
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        预测障碍物未来分布
        
        Returns:
            mean_predictions: 均值预测 [T_pred, num_obstacles, 2]
            std_predictions: 标准差 (不确定性) [T_pred, num_obstacles, 2]
            confidence: 预测置信度 (0-1)
        """
        T_obs, num_obstacles, _ = observations.shape
        
        # 计算各障碍物的速度估计
        if T_obs >= 2:
            velocities = observations[-1] - observations[-2]  # [num_obstacles, 2]
            # 速度平滑
            if T_obs >= 3:
                velocities = (observations[-1] - observations[-3]) / 2
        else:
            velocities = torch.zeros(num_obstacles, 2, device=self.device)
        
        # 扩散采样：生成多条可能轨迹
        all_samples = []
        
        for _ in range(self.config.num_obstacle_samples):
            # 初始化为噪声
            sample = torch.randn(prediction_steps, num_obstacles, 2, device=self.device)
            
            # 目标：常速度预测 + 随机扰动
            target = torch.zeros_like(sample)
            for t in range(prediction_steps):
                # 基于速度的预测
                target[t] = observations[-1] + (t + 1) * velocities
                # 添加位置相关的不确定性（离当前越远越不确定）
                target[t] += torch.randn_like(target[t]) * 0.05 * (t + 1) / prediction_steps
            
            # 扩散去噪：从噪声收敛到预测
            for step in reversed(range(self.num_steps)):
                alpha_bar = self.alpha_bars[step]
                alpha = self.alphas[step]
                
                # 引导去噪
                noise_estimate = (sample - target * alpha_bar.sqrt()) / (1 - alpha_bar).sqrt().clamp(min=1e-6)
                
                if step > 0:
                    noise = torch.randn_like(sample) * self.betas[step].sqrt()
                    sample = (1 / alpha.sqrt()) * (sample - (1 - alpha) / (1 - alpha_bar).sqrt().clamp(min=1e-6) * noise_estimate)
                    sample = sample + noise
                else:
                    sample = (1 / alpha.sqrt()) * (sample - (1 - alpha) / (1 - alpha_bar).sqrt().clamp(min=1e-6) * noise_estimate)
                
                # 引导强度
                guide = 0.1 * (1 - step / self.num_steps)
                sample = sample + guide * (target - sample)
            
            all_samples.append(sample)
        
        # 计算均值和标准差
        samples_stack = torch.stack(all_samples, dim=0)  # [num_samples, T_pred, num_obs, 2]
        mean_predictions = samples_stack.mean(dim=0)
        std_predictions = samples_stack.std(dim=0)
        
        # 计算置信度（基于预测方差）
        avg_std = std_predictions.mean().item()
        confidence = 1.0 / (1.0 + avg_std * 10)  # 方差越小置信度越高
        
        return mean_predictions, std_predictions, confidence


# ============================================================================
# 中层: 自适应速度控制器
# ============================================================================

class AdaptiveSpeedController:
    """
    中层: 自适应速度控制器
    
    根据预测置信度和安全状态调整速度模式
    """
    
    def __init__(self, config: MPCConfig):
        self.config = config
        self.current_mode = SpeedMode.PROBE
        self.mode_history = []
        self.stability_counter = 0
    
    def update(
        self,
        prediction_confidence: float,
        min_obstacle_distance: float,
        trajectory_variance: float
    ) -> Tuple[SpeedMode, float]:
        """
        更新速度模式
        
        Args:
            prediction_confidence: 预测置信度 (0-1)
            min_obstacle_distance: 到最近障碍物的距离
            trajectory_variance: 规划轨迹的方差（稳定性）
            
        Returns:
            mode: 当前速度模式
            max_speed: 最大允许速度
        """
        prev_mode = self.current_mode
        
        # 安全检查：太近障碍物强制试探
        if min_obstacle_distance < self.config.safety_margin:
            self.current_mode = SpeedMode.PROBE
            self.stability_counter = 0
        # 根据置信度选择模式
        elif prediction_confidence < self.config.confidence_threshold_low:
            self.current_mode = SpeedMode.PROBE
            self.stability_counter = 0
        elif prediction_confidence > self.config.confidence_threshold_high:
            # 需要连续高置信度才进入冲刺
            self.stability_counter += 1
            if self.stability_counter >= 5:  # 连续5个周期
                self.current_mode = SpeedMode.SPRINT
            else:
                self.current_mode = SpeedMode.TRACK
        else:
            self.current_mode = SpeedMode.TRACK
            self.stability_counter = max(0, self.stability_counter - 1)
        
        # 轨迹不稳定时降级
        if trajectory_variance > 0.1 and self.current_mode == SpeedMode.SPRINT:
            self.current_mode = SpeedMode.TRACK
            self.stability_counter = 0
        
        # 记录历史
        self.mode_history.append(self.current_mode)
        if len(self.mode_history) > 100:
            self.mode_history.pop(0)
        
        # 返回速度
        if self.current_mode == SpeedMode.PROBE:
            max_speed = self.config.speed_probe
        elif self.current_mode == SpeedMode.TRACK:
            max_speed = self.config.speed_track
        else:
            max_speed = self.config.speed_sprint
        
        return self.current_mode, max_speed
    
    def get_mode_description(self) -> str:
        """获取模式描述"""
        if self.current_mode == SpeedMode.PROBE:
            return "🐢 PROBE (慢速试探，优先避障)"
        elif self.current_mode == SpeedMode.TRACK:
            return "🚶 TRACK (中速跟踪，沿轨迹前进)"
        else:
            return "🏃 SPRINT (快速冲刺，直奔目标)"


# ============================================================================
# 下层: 轨迹规划扩散模型
# ============================================================================

class TrajectoryPlanningDiffusion:
    """
    下层: 轨迹规划扩散模型
    
    从噪声中采样多条轨迹，通过障碍物分布引导收敛到最优
    """
    
    def __init__(self, config: MPCConfig, device='cpu'):
        self.config = config
        self.device = device
        
        self.num_steps = config.diffusion_steps
        self.betas = torch.linspace(1e-4, 0.02, self.num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def plan(
        self,
        current_pos: torch.Tensor,        # [2]
        goal_pos: torch.Tensor,           # [2]
        obstacle_means: torch.Tensor,     # [T_pred, num_obstacles, 2]
        obstacle_stds: torch.Tensor,      # [T_pred, num_obstacles, 2]
        max_speed: float
    ) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
        """
        规划轨迹
        
        Returns:
            best_trajectory: 最优轨迹 [T, 2]
            all_samples: 所有采样轨迹（用于可视化）
            trajectory_variance: 轨迹方差
        """
        T = self.config.planning_horizon
        
        # 直线目标轨迹
        target_traj = torch.zeros(T, 2, device=self.device)
        for i in range(T):
            t = i / (T - 1)
            target_traj[i] = current_pos + t * (goal_pos - current_pos)
        
        # 扩散采样多条轨迹
        all_samples = []
        all_scores = []
        
        for _ in range(self.config.num_trajectory_samples):
            # 从噪声开始
            sample = torch.randn(T, 2, device=self.device) * 0.3
            sample = sample + target_traj  # 以直线为中心
            sample[0] = current_pos
            sample[-1] = goal_pos
            
            # 扩散去噪
            for step in reversed(range(self.num_steps)):
                alpha_bar = self.alpha_bars[step]
                alpha = self.alphas[step]
                
                # 基础去噪
                noise_estimate = (sample - target_traj * alpha_bar.sqrt()) / (1 - alpha_bar).sqrt().clamp(min=1e-6)
                
                if step > 0:
                    noise = torch.randn_like(sample) * self.betas[step].sqrt() * 0.5
                    sample = (1 / alpha.sqrt()) * (sample - (1 - alpha) / (1 - alpha_bar).sqrt().clamp(min=1e-6) * noise_estimate)
                    sample = sample + noise
                else:
                    sample = (1 / alpha.sqrt()) * (sample - (1 - alpha) / (1 - alpha_bar).sqrt().clamp(min=1e-6) * noise_estimate)
                
                # 障碍物分布引导（关键！）
                obstacle_guidance = self._compute_obstacle_guidance(
                    sample, obstacle_means, obstacle_stds
                )
                guide_strength = 0.05 * (1 - step / self.num_steps)
                sample = sample + guide_strength * obstacle_guidance
                
                # 目标引导
                target_guide = 0.1 * (1 - step / self.num_steps)
                sample = sample + target_guide * (target_traj - sample)
                
                # 固定端点
                sample[0] = current_pos
                sample[-1] = goal_pos
            
            # 速度约束投影
            sample = self._project_speed_constraint(sample, current_pos, goal_pos, max_speed)
            
            # 评估轨迹
            score = self._evaluate_trajectory(sample, obstacle_means, obstacle_stds, goal_pos)
            
            all_samples.append(sample)
            all_scores.append(score)
        
        # 选择最优轨迹
        best_idx = np.argmax(all_scores)
        best_trajectory = all_samples[best_idx]
        
        # 计算轨迹方差（稳定性指标）
        samples_stack = torch.stack(all_samples, dim=0)
        trajectory_variance = samples_stack.std(dim=0).mean().item()
        
        return best_trajectory, all_samples, trajectory_variance
    
    def _compute_obstacle_guidance(
        self,
        trajectory: torch.Tensor,
        obstacle_means: torch.Tensor,
        obstacle_stds: torch.Tensor
    ) -> torch.Tensor:
        """
        计算障碍物分布的引导力
        
        考虑障碍物的不确定性：不确定性越大，避让范围越大
        """
        T = len(trajectory)
        T_obs = len(obstacle_means)
        guidance = torch.zeros_like(trajectory)
        
        for t in range(1, T-1):
            # 插值到对应时刻的障碍物位置
            obs_t = min(int(t / T * T_obs), T_obs - 1)
            
            for obs_idx in range(obstacle_means.shape[1]):
                obs_pos = obstacle_means[obs_t, obs_idx]
                obs_std = obstacle_stds[obs_t, obs_idx].mean()
                
                # 根据不确定性调整避让半径
                avoid_radius = 0.1 + obs_std.item() * 2  # 不确定性越大，避让越远
                
                diff = trajectory[t] - obs_pos
                dist = torch.norm(diff)
                
                if dist < avoid_radius and dist > 0.01:
                    # 推离力：距离越近力越大
                    force = (avoid_radius - dist) / avoid_radius
                    guidance[t] += force * diff / dist
        
        return guidance
    
    def _project_speed_constraint(
        self,
        trajectory: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        max_speed: float
    ) -> torch.Tensor:
        """投影到速度约束空间"""
        T = len(trajectory)
        proj = trajectory.clone()
        proj[0] = start
        
        # 前向约束
        for i in range(1, T):
            vel = proj[i] - proj[i-1]
            speed = torch.norm(vel)
            if speed > max_speed:
                proj[i] = proj[i-1] + vel * max_speed / speed
        
        # 后向约束（确保能到达终点）
        proj[-1] = goal
        for i in range(T-2, 0, -1):
            vel = proj[i+1] - proj[i]
            speed = torch.norm(vel)
            if speed > max_speed:
                proj[i] = proj[i+1] - vel * max_speed / speed
        
        proj[0] = start
        proj[-1] = goal
        
        return proj
    
    def _evaluate_trajectory(
        self,
        trajectory: torch.Tensor,
        obstacle_means: torch.Tensor,
        obstacle_stds: torch.Tensor,
        goal: torch.Tensor
    ) -> float:
        """评估轨迹质量"""
        T = len(trajectory)
        T_obs = len(obstacle_means)
        
        # 障碍物距离分数（考虑不确定性）
        min_clearance = float('inf')
        for t in range(T):
            obs_t = min(int(t / T * T_obs), T_obs - 1)
            for obs_idx in range(obstacle_means.shape[1]):
                obs_pos = obstacle_means[obs_t, obs_idx]
                obs_std = obstacle_stds[obs_t, obs_idx].mean().item()
                
                dist = torch.norm(trajectory[t] - obs_pos).item()
                # 距离减去不确定性范围
                effective_clearance = dist - obs_std * 2
                min_clearance = min(min_clearance, effective_clearance)
        
        # 路径长度惩罚
        path_length = torch.norm(trajectory[1:] - trajectory[:-1], dim=-1).sum().item()
        
        # 到达目标奖励
        goal_dist = torch.norm(trajectory[-1] - goal).item()
        
        # 综合分数
        score = min_clearance * 10 - path_length * 0.1 - goal_dist * 5
        
        return score


# ============================================================================
# 完整MPC控制器
# ============================================================================

class RealtimeHierarchicalMPC:
    """
    实时层级扩散MPC控制器
    
    整合上、中、下三层，实现实时控制
    """
    
    def __init__(self, config: MPCConfig = None, device='cpu'):
        self.config = config or MPCConfig()
        self.device = device
        
        # 三层模块
        self.obstacle_predictor = ObstaclePredictionDiffusion(self.config, device)
        self.speed_controller = AdaptiveSpeedController(self.config)
        self.trajectory_planner = TrajectoryPlanningDiffusion(self.config, device)
        
        # 状态
        self.current_pos = None
        self.goal_pos = None
        self.obstacle_history = []
        self.trajectory_history = []
        
        # 诊断信息
        self.last_info = {}
    
    def reset(self, start: torch.Tensor, goal: torch.Tensor):
        """重置控制器"""
        self.current_pos = start.clone()
        self.goal_pos = goal.clone()
        self.obstacle_history = []
        self.trajectory_history = []
        self.speed_controller.stability_counter = 0
        self.speed_controller.current_mode = SpeedMode.PROBE
    
    def observe_obstacles(self, obstacle_positions: torch.Tensor):
        """
        观测障碍物位置
        
        Args:
            obstacle_positions: [num_obstacles, 2] 当前时刻各障碍物位置
        """
        self.obstacle_history.append(obstacle_positions.clone())
        # 保留最近8帧
        if len(self.obstacle_history) > 8:
            self.obstacle_history.pop(0)
    
    def step(self) -> Tuple[torch.Tensor, Dict]:
        """
        执行一步MPC控制
        
        Returns:
            control: 下一步速度控制 [2]
            info: 诊断信息
        """
        info = {}
        
        # 检查是否有足够观测
        if len(self.obstacle_history) < 2:
            info['error'] = 'Not enough obstacle observations'
            return torch.zeros(2, device=self.device), info
        
        # ========== 上层: 障碍物预测 ==========
        obs_tensor = torch.stack(self.obstacle_history, dim=0)  # [T_obs, num_obs, 2]
        
        obstacle_means, obstacle_stds, prediction_confidence = \
            self.obstacle_predictor.predict(obs_tensor, prediction_steps=12)
        
        info['prediction_confidence'] = prediction_confidence
        info['obstacle_means'] = obstacle_means
        info['obstacle_stds'] = obstacle_stds
        
        # ========== 中层: 速度控制 ==========
        # 计算到最近障碍物的距离
        current_obstacles = self.obstacle_history[-1]
        distances = torch.norm(current_obstacles - self.current_pos, dim=-1)
        min_obstacle_dist = distances.min().item()
        
        # 获取上一次轨迹方差（如果有）
        last_variance = self.last_info.get('trajectory_variance', 0.5)
        
        speed_mode, max_speed = self.speed_controller.update(
            prediction_confidence,
            min_obstacle_dist,
            last_variance
        )
        
        info['speed_mode'] = speed_mode
        info['max_speed'] = max_speed
        info['min_obstacle_dist'] = min_obstacle_dist
        
        # ========== 下层: 轨迹规划 ==========
        best_trajectory, all_samples, trajectory_variance = \
            self.trajectory_planner.plan(
                self.current_pos,
                self.goal_pos,
                obstacle_means,
                obstacle_stds,
                max_speed
            )
        
        info['trajectory'] = best_trajectory
        info['all_trajectory_samples'] = all_samples
        info['trajectory_variance'] = trajectory_variance
        
        # 提取控制指令（下一步速度）
        if len(best_trajectory) >= 2:
            control = best_trajectory[1] - best_trajectory[0]
            # 限制速度
            speed = torch.norm(control)
            if speed > max_speed:
                control = control * max_speed / speed
        else:
            control = torch.zeros(2, device=self.device)
        
        info['control'] = control
        
        # 更新状态
        self.current_pos = self.current_pos + control
        self.trajectory_history.append(self.current_pos.clone())
        self.last_info = info
        
        return control, info
    
    def is_goal_reached(self, threshold: float = 0.05) -> bool:
        """检查是否到达目标"""
        return torch.norm(self.current_pos - self.goal_pos).item() < threshold


# ============================================================================
# 便捷函数
# ============================================================================

def create_mpc_controller(device='cpu') -> RealtimeHierarchicalMPC:
    """创建MPC控制器"""
    config = MPCConfig()
    return RealtimeHierarchicalMPC(config, device)

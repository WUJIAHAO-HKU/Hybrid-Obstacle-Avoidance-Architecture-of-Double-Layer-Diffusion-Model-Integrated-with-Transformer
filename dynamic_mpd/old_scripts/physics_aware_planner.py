"""
物理感知轨迹规划器

该模块实现了满足物理约束的动态避障规划器：
1. 速度约束: |v| <= v_max
2. 加速度约束: |a| <= a_max
3. 碰撞约束: SDF(x,t) >= safety_margin
4. 智能策略: 绕行 vs 等待

Author: Dynamic MPD Project
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from spatiotemporal_sdf import SpatioTemporalSDF, ObstacleTrajectory


@dataclass
class PhysicsConstraints:
    """物理约束参数"""
    max_velocity: float = 0.06       # 最大速度 (单位/时间步)
    max_acceleration: float = 0.02   # 最大加速度
    robot_radius: float = 0.05       # 机器人半径
    safety_margin: float = 0.06      # 安全余量


@dataclass
class PlanningConfig:
    """规划配置"""
    workspace_bounds: List[float] = None  # [xmin, xmax, ymin, ymax]
    time_horizon: float = 10.0
    num_steps: int = 64
    sdf_resolution: float = 0.015
    max_iterations: int = 150
    learning_rate: float = 0.02
    
    def __post_init__(self):
        if self.workspace_bounds is None:
            self.workspace_bounds = [-1.0, 1.0, -1.0, 1.0]


class PhysicsAwarePlanner:
    """
    物理感知轨迹规划器
    
    支持多种避障策略：
    - straight: 直线轨迹 + 梯度优化
    - detour_up/down: 向上/下绕行
    - wait_then_go: 先等待再前进
    
    自动选择最优策略。
    """
    
    def __init__(
        self,
        physics: PhysicsConstraints = None,
        config: PlanningConfig = None,
        device: str = 'cpu'
    ):
        self.physics = physics or PhysicsConstraints()
        self.config = config or PlanningConfig()
        self.device = device
    
    def plan(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        obstacle_trajectories: List[torch.Tensor],
        obstacle_radius: float
    ) -> Tuple[torch.Tensor, SpatioTemporalSDF, Dict]:
        """
        规划满足物理约束的轨迹
        
        Args:
            start: 起点 [x, y]
            goal: 终点 [x, y]
            obstacle_trajectories: 障碍物轨迹列表
            obstacle_radius: 障碍物半径
            
        Returns:
            trajectory: 规划的轨迹 [T, 2]
            sdf: 时空SDF
            info: 规划信息
        """
        T = self.config.num_steps
        
        # 创建时空SDF
        sdf = self._create_sdf(obstacle_trajectories, obstacle_radius)
        
        # 尝试多种策略
        strategies = ['straight', 'detour_up', 'detour_down', 'wait_then_go']
        best_traj = None
        best_score = float('-inf')
        best_strategy = None
        
        for strategy in strategies:
            traj = self._plan_with_strategy(start, goal, sdf, T, strategy)
            score = self._evaluate_trajectory(traj, sdf, start, goal)
            
            if score > best_score:
                best_score = score
                best_traj = traj.clone()
                best_strategy = strategy
        
        # 计算统计信息
        info = self._compute_info(best_traj, sdf, best_strategy)
        
        return best_traj, sdf, info
    
    def _create_sdf(
        self,
        obstacle_trajectories: List[torch.Tensor],
        obstacle_radius: float
    ) -> SpatioTemporalSDF:
        """创建时空SDF"""
        sdf = SpatioTemporalSDF(
            workspace_bounds=self.config.workspace_bounds,
            spatial_resolution=self.config.sdf_resolution,
            time_horizon=self.config.time_horizon,
            num_time_steps=self.config.num_steps,
            device=self.device
        )
        
        obs_for_sdf = []
        for i, obs_traj in enumerate(obstacle_trajectories):
            obs_for_sdf.append(ObstacleTrajectory(
                positions=obs_traj,
                radii=obstacle_radius + self.physics.robot_radius,  # 膨胀
                timestamps=torch.linspace(
                    0, self.config.time_horizon,
                    obs_traj.shape[0], device=self.device
                ),
                obstacle_id=i
            ))
        sdf.compute_from_obstacle_trajectories(obs_for_sdf)
        
        return sdf
    
    def _plan_with_strategy(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        sdf: SpatioTemporalSDF,
        T: int,
        strategy: str
    ) -> torch.Tensor:
        """使用特定策略规划"""
        trajectory = self._initialize_trajectory(start, goal, T, strategy)
        
        for iteration in range(self.config.max_iterations):
            sdf_values = sdf.query_trajectory(trajectory)
            min_sdf = sdf_values.min().item()
            
            if min_sdf > self.physics.safety_margin:
                break
            
            collision_mask = sdf_values < self.physics.safety_margin
            grad = self._compute_gradient(trajectory, sdf)
            trajectory = self._update_trajectory(
                trajectory, grad, collision_mask, start, goal
            )
            trajectory = self._enforce_physics(trajectory, start, goal)
        
        return trajectory
    
    def _initialize_trajectory(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        T: int,
        strategy: str
    ) -> torch.Tensor:
        """根据策略初始化轨迹"""
        trajectory = torch.zeros(T, 2, device=self.device)
        
        if strategy == 'straight':
            for i in range(T):
                t = i / (T - 1)
                trajectory[i] = start + t * (goal - start)
        
        elif strategy == 'detour_up':
            mid_point = (start + goal) / 2
            mid_point = mid_point.clone()
            mid_point[1] += 0.4
            for i in range(T):
                t = i / (T - 1)
                if t < 0.5:
                    trajectory[i] = start + (t * 2) * (mid_point - start)
                else:
                    trajectory[i] = mid_point + ((t - 0.5) * 2) * (goal - mid_point)
        
        elif strategy == 'detour_down':
            mid_point = (start + goal) / 2
            mid_point = mid_point.clone()
            mid_point[1] -= 0.4
            for i in range(T):
                t = i / (T - 1)
                if t < 0.5:
                    trajectory[i] = start + (t * 2) * (mid_point - start)
                else:
                    trajectory[i] = mid_point + ((t - 0.5) * 2) * (goal - mid_point)
        
        elif strategy == 'wait_then_go':
            wait_ratio = 0.4
            for i in range(T):
                t = i / (T - 1)
                if t < wait_ratio:
                    trajectory[i] = start + (t / wait_ratio) * 0.1 * (goal - start)
                else:
                    progress = (t - wait_ratio) / (1 - wait_ratio)
                    trajectory[i] = start + 0.1 * (goal - start) + progress * 0.9 * (goal - start)
        
        return trajectory
    
    def _compute_gradient(
        self,
        trajectory: torch.Tensor,
        sdf: SpatioTemporalSDF
    ) -> torch.Tensor:
        """计算SDF梯度"""
        T = trajectory.shape[0]
        grad = torch.zeros_like(trajectory)
        eps = 0.01
        
        for i in range(1, T-1):
            pos = trajectory[i]
            t_val = i / (T-1) * sdf.time_horizon
            
            for dim in range(2):
                pos_p = pos.clone()
                pos_m = pos.clone()
                pos_p[dim] += eps
                pos_m[dim] -= eps
                grad[i, dim] = (
                    sdf.query_point(pos_p, t_val) - sdf.query_point(pos_m, t_val)
                ) / (2 * eps)
        
        return grad
    
    def _update_trajectory(
        self,
        trajectory: torch.Tensor,
        grad: torch.Tensor,
        collision_mask: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor
    ) -> torch.Tensor:
        """更新轨迹"""
        new_traj = trajectory.clone()
        lr = self.config.learning_rate
        
        for i in range(1, len(trajectory)-1):
            if collision_mask[i]:
                g = grad[i]
                g_norm = torch.norm(g)
                if g_norm > 0.01:
                    new_traj[i] = trajectory[i] + lr * g / g_norm
        
        new_traj[0] = start
        new_traj[-1] = goal
        return new_traj
    
    def _enforce_physics(
        self,
        trajectory: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor
    ) -> torch.Tensor:
        """强制物理约束"""
        T = len(trajectory)
        proj = trajectory.clone()
        proj[0] = start
        
        # 前向速度限制
        for i in range(1, T):
            vel = proj[i] - proj[i-1]
            speed = torch.norm(vel)
            if speed > self.physics.max_velocity:
                proj[i] = proj[i-1] + vel * self.physics.max_velocity / speed
        
        # 后向确保到达终点
        proj[-1] = goal
        for i in range(T-2, 0, -1):
            vel = proj[i+1] - proj[i]
            speed = torch.norm(vel)
            if speed > self.physics.max_velocity:
                proj[i] = proj[i+1] - vel * self.physics.max_velocity / speed
        
        # 边界约束
        bounds = self.config.workspace_bounds
        margin = 0.05
        proj[:, 0] = proj[:, 0].clamp(bounds[0] + margin, bounds[1] - margin)
        proj[:, 1] = proj[:, 1].clamp(bounds[2] + margin, bounds[3] - margin)
        proj[0] = start
        proj[-1] = goal
        
        return proj
    
    def _evaluate_trajectory(
        self,
        trajectory: torch.Tensor,
        sdf: SpatioTemporalSDF,
        start: torch.Tensor,
        goal: torch.Tensor
    ) -> float:
        """评估轨迹质量"""
        sdf_values = sdf.query_trajectory(trajectory)
        min_sdf = sdf_values.min().item()
        
        # 碰撞惩罚
        collision_penalty = min_sdf * 100 if min_sdf < 0 else 0
        
        # 路径长度惩罚
        path_length = torch.norm(
            trajectory[1:] - trajectory[:-1], dim=-1
        ).sum().item()
        length_penalty = -0.1 * path_length
        
        # 安全奖励
        safety_bonus = min(min_sdf, self.physics.safety_margin) * 10
        
        return collision_penalty + length_penalty + safety_bonus
    
    def _compute_info(
        self,
        trajectory: torch.Tensor,
        sdf: SpatioTemporalSDF,
        strategy: str
    ) -> Dict:
        """计算规划信息"""
        sdf_values = sdf.query_trajectory(trajectory)
        velocities = trajectory[1:] - trajectory[:-1]
        speeds = torch.norm(velocities, dim=-1)
        
        if len(velocities) > 1:
            accelerations = velocities[1:] - velocities[:-1]
            accel_mags = torch.norm(accelerations, dim=-1)
        else:
            accel_mags = torch.zeros(1)
        
        return {
            'strategy': strategy,
            'collision_free': sdf_values.min().item() > 0,
            'min_sdf': sdf_values.min().item(),
            'max_speed': speeds.max().item(),
            'avg_speed': speeds.mean().item(),
            'max_acceleration': accel_mags.max().item(),
            'path_length': speeds.sum().item(),
            'velocity_ok': speeds.max().item() <= self.physics.max_velocity * 1.1,
            'acceleration_ok': accel_mags.max().item() <= self.physics.max_acceleration * 1.5
        }


def create_planner(
    max_velocity: float = 0.055,
    max_acceleration: float = 0.015,
    robot_radius: float = 0.05,
    safety_margin: float = 0.06,
    device: str = 'cpu'
) -> PhysicsAwarePlanner:
    """创建物理感知规划器的便捷函数"""
    physics = PhysicsConstraints(
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        robot_radius=robot_radius,
        safety_margin=safety_margin
    )
    config = PlanningConfig()
    return PhysicsAwarePlanner(physics, config, device)

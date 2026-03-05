"""
障碍物轨迹预测器

提供多种预测方法：
1. 线性预测 (Linear) - 简单外推
2. 卡尔曼滤波 (Kalman) - 考虑噪声
3. 扩散模型预测 (Diffusion) - 后续实现

Author: Dynamic MPD Project
Date: 2026-01-22
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .spatiotemporal_sdf import ObstacleTrajectory


@dataclass
class ObstacleState:
    """障碍物状态"""
    position: torch.Tensor   # [2] 或 [3]
    velocity: torch.Tensor   # [2] 或 [3]
    radius: float
    obstacle_id: int = 0
    timestamp: float = 0.0


class ObstaclePredictorBase(ABC):
    """障碍物预测器基类"""
    
    def __init__(
        self,
        prediction_horizon: int = 100,
        dt: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            prediction_horizon: 预测步数
            dt: 时间步长 (秒)
            device: 计算设备
        """
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        self.device = device
        
    @abstractmethod
    def predict(
        self,
        obstacle_states: List[ObstacleState],
    ) -> List[ObstacleTrajectory]:
        """
        预测障碍物未来轨迹
        
        Args:
            obstacle_states: 当前障碍物状态列表
            
        Returns:
            predicted_trajectories: 预测的轨迹列表
        """
        pass
    
    @abstractmethod
    def update(
        self,
        obstacle_states: List[ObstacleState],
    ):
        """
        用新观测更新预测器状态
        
        Args:
            obstacle_states: 最新观测的障碍物状态
        """
        pass


class LinearObstaclePredictor(ObstaclePredictorBase):
    """
    线性障碍物预测器
    
    使用恒定速度模型进行简单线性外推：
    x(t+k) = x(t) + k * v(t) * dt
    
    这是最简单的基准预测方法。
    """
    
    def __init__(
        self,
        prediction_horizon: int = 100,
        dt: float = 0.1,
        velocity_decay: float = 1.0,  # 速度衰减因子 (1.0 = 无衰减)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(prediction_horizon, dt, device)
        self.velocity_decay = velocity_decay
        
        # 历史状态缓存 (用于速度估计)
        self.history: dict = {}  # obstacle_id -> List[ObstacleState]
        self.history_length = 5  # 保留的历史长度
        
    def predict(
        self,
        obstacle_states: List[ObstacleState],
    ) -> List[ObstacleTrajectory]:
        """
        线性预测障碍物轨迹
        
        Args:
            obstacle_states: 当前障碍物状态列表
            
        Returns:
            predicted_trajectories: 预测轨迹列表
        """
        predicted_trajectories = []
        
        for obs_state in obstacle_states:
            # 获取或估计速度
            velocity = self._get_velocity(obs_state)
            
            # 生成预测轨迹
            positions = []
            current_pos = obs_state.position.clone()
            current_vel = velocity.clone()
            
            for k in range(self.prediction_horizon):
                positions.append(current_pos.clone())
                
                # 线性外推
                current_pos = current_pos + current_vel * self.dt
                
                # 可选：速度衰减
                current_vel = current_vel * self.velocity_decay
                
            positions = torch.stack(positions)  # [T, dim]
            timestamps = torch.arange(
                self.prediction_horizon, device=self.device
            ) * self.dt
            
            trajectory = ObstacleTrajectory(
                positions=positions,
                radii=obs_state.radius,
                timestamps=timestamps,
                obstacle_id=obs_state.obstacle_id,
            )
            predicted_trajectories.append(trajectory)
            
        return predicted_trajectories
    
    def _get_velocity(self, obs_state: ObstacleState) -> torch.Tensor:
        """
        获取或估计障碍物速度
        
        如果状态中已有速度，直接使用；
        否则从历史数据估计。
        """
        # 如果状态中有速度信息
        if obs_state.velocity is not None and torch.any(obs_state.velocity != 0):
            return obs_state.velocity
            
        # 从历史估计速度
        obs_id = obs_state.obstacle_id
        if obs_id in self.history and len(self.history[obs_id]) >= 2:
            prev_state = self.history[obs_id][-1]
            dt = obs_state.timestamp - prev_state.timestamp
            if dt > 0:
                velocity = (obs_state.position - prev_state.position) / dt
                return velocity
                
        # 默认返回零速度
        dim = obs_state.position.shape[0]
        return torch.zeros(dim, device=self.device)
    
    def update(self, obstacle_states: List[ObstacleState]):
        """
        更新历史状态
        """
        for obs_state in obstacle_states:
            obs_id = obs_state.obstacle_id
            
            if obs_id not in self.history:
                self.history[obs_id] = []
                
            self.history[obs_id].append(obs_state)
            
            # 保持历史长度
            if len(self.history[obs_id]) > self.history_length:
                self.history[obs_id] = self.history[obs_id][-self.history_length:]


class KalmanObstaclePredictor(ObstaclePredictorBase):
    """
    卡尔曼滤波障碍物预测器
    
    使用恒定速度模型的卡尔曼滤波进行预测，
    能够处理观测噪声和提供不确定性估计。
    
    状态向量: [x, y, vx, vy] (2D) 或 [x, y, z, vx, vy, vz] (3D)
    """
    
    def __init__(
        self,
        prediction_horizon: int = 100,
        dt: float = 0.1,
        process_noise: float = 0.1,
        measurement_noise: float = 0.05,
        dim: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(prediction_horizon, dt, device)
        self.dim = dim
        self.state_dim = dim * 2  # 位置 + 速度
        
        # 卡尔曼滤波参数
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # 状态转移矩阵 F
        self.F = self._create_transition_matrix()
        
        # 观测矩阵 H (只观测位置)
        self.H = torch.zeros(dim, self.state_dim, device=device)
        self.H[:dim, :dim] = torch.eye(dim, device=device)
        
        # 过程噪声协方差 Q
        self.Q = torch.eye(self.state_dim, device=device) * process_noise
        
        # 观测噪声协方差 R
        self.R = torch.eye(dim, device=device) * measurement_noise
        
        # 每个障碍物的滤波器状态
        self.filters: dict = {}  # obstacle_id -> (state, covariance)
        
    def _create_transition_matrix(self) -> torch.Tensor:
        """创建状态转移矩阵"""
        F = torch.eye(self.state_dim, device=self.device)
        # 位置更新: x' = x + v * dt
        for i in range(self.dim):
            F[i, self.dim + i] = self.dt
        return F
    
    def _initialize_filter(self, obs_state: ObstacleState) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化单个障碍物的卡尔曼滤波器"""
        state = torch.zeros(self.state_dim, device=self.device)
        state[:self.dim] = obs_state.position
        
        if obs_state.velocity is not None:
            state[self.dim:] = obs_state.velocity
            
        covariance = torch.eye(self.state_dim, device=self.device) * 1.0
        
        return state, covariance
    
    def _predict_step(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """卡尔曼滤波预测步"""
        # 状态预测
        state_pred = self.F @ state
        
        # 协方差预测
        cov_pred = self.F @ covariance @ self.F.T + self.Q
        
        return state_pred, cov_pred
    
    def _update_step(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor,
        measurement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """卡尔曼滤波更新步"""
        # 新息 (Innovation)
        y = measurement - self.H @ state
        
        # 新息协方差
        S = self.H @ covariance @ self.H.T + self.R
        
        # 卡尔曼增益
        K = covariance @ self.H.T @ torch.inverse(S)
        
        # 状态更新
        state_new = state + K @ y
        
        # 协方差更新
        I = torch.eye(self.state_dim, device=self.device)
        cov_new = (I - K @ self.H) @ covariance
        
        return state_new, cov_new
    
    def predict(
        self,
        obstacle_states: List[ObstacleState],
    ) -> List[ObstacleTrajectory]:
        """
        卡尔曼滤波预测
        """
        predicted_trajectories = []
        
        for obs_state in obstacle_states:
            obs_id = obs_state.obstacle_id
            
            # 获取或初始化滤波器
            if obs_id not in self.filters:
                state, cov = self._initialize_filter(obs_state)
            else:
                state, cov = self.filters[obs_id]
                
            # 生成预测轨迹
            positions = []
            current_state = state.clone()
            current_cov = cov.clone()
            
            for k in range(self.prediction_horizon):
                positions.append(current_state[:self.dim].clone())
                current_state, current_cov = self._predict_step(current_state, current_cov)
                
            positions = torch.stack(positions)
            timestamps = torch.arange(
                self.prediction_horizon, device=self.device
            ) * self.dt
            
            trajectory = ObstacleTrajectory(
                positions=positions,
                radii=obs_state.radius,
                timestamps=timestamps,
                obstacle_id=obs_id,
            )
            predicted_trajectories.append(trajectory)
            
        return predicted_trajectories
    
    def update(self, obstacle_states: List[ObstacleState]):
        """
        用新观测更新滤波器
        """
        for obs_state in obstacle_states:
            obs_id = obs_state.obstacle_id
            
            if obs_id not in self.filters:
                state, cov = self._initialize_filter(obs_state)
            else:
                state, cov = self.filters[obs_id]
                # 先预测
                state, cov = self._predict_step(state, cov)
                # 再更新
                state, cov = self._update_step(state, cov, obs_state.position)
                
            self.filters[obs_id] = (state, cov)


class ConstantVelocityPredictor(LinearObstaclePredictor):
    """
    恒速预测器 (LinearObstaclePredictor的别名)
    """
    pass


def create_predictor(
    predictor_type: str = "linear",
    **kwargs
) -> ObstaclePredictorBase:
    """
    工厂函数：创建障碍物预测器
    
    Args:
        predictor_type: 预测器类型 ("linear", "kalman", "diffusion")
        **kwargs: 预测器参数
        
    Returns:
        predictor: 障碍物预测器实例
    """
    predictors = {
        "linear": LinearObstaclePredictor,
        "kalman": KalmanObstaclePredictor,
        "constant_velocity": ConstantVelocityPredictor,
    }
    
    if predictor_type not in predictors:
        raise ValueError(f"Unknown predictor type: {predictor_type}. "
                        f"Available: {list(predictors.keys())}")
        
    return predictors[predictor_type](**kwargs)

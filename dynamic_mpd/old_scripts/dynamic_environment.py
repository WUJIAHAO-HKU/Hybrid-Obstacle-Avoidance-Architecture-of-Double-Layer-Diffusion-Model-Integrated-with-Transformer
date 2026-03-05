"""
动态环境定义

在静态环境基础上添加动态障碍物支持。

Author: Dynamic MPD Project
Date: 2026-01-22
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class MotionType(Enum):
    """动态障碍物运动类型"""
    STATIC = "static"
    LINEAR = "linear"          # 直线运动
    CIRCULAR = "circular"      # 圆周运动
    RANDOM_WALK = "random_walk"  # 随机游走
    SINUSOIDAL = "sinusoidal"  # 正弦运动


@dataclass
class DynamicObstacle:
    """
    动态障碍物定义
    
    Attributes:
        obstacle_id: 障碍物唯一ID
        initial_position: 初始位置 [2] 或 [3]
        radius: 障碍物半径
        motion_type: 运动类型
        velocity: 速度向量 (用于LINEAR)
        angular_velocity: 角速度 (用于CIRCULAR)
        center: 圆周运动中心 (用于CIRCULAR)
        amplitude: 振幅 (用于SINUSOIDAL)
        frequency: 频率 (用于SINUSOIDAL)
        noise_std: 噪声标准差 (用于RANDOM_WALK)
    """
    obstacle_id: int
    initial_position: torch.Tensor
    radius: float = 0.1
    motion_type: MotionType = MotionType.LINEAR
    
    # LINEAR 参数
    velocity: Optional[torch.Tensor] = None
    
    # CIRCULAR 参数
    angular_velocity: float = 0.5
    orbit_radius: float = 0.3
    center: Optional[torch.Tensor] = None
    initial_angle: float = 0.0
    
    # SINUSOIDAL 参数
    amplitude: float = 0.2
    frequency: float = 0.5
    direction: Optional[torch.Tensor] = None
    
    # RANDOM_WALK 参数
    noise_std: float = 0.05
    
    # 内部状态
    _current_position: torch.Tensor = field(default=None, repr=False)
    _current_velocity: torch.Tensor = field(default=None, repr=False)
    _time: float = field(default=0.0, repr=False)
    
    def __post_init__(self):
        """初始化内部状态"""
        self._current_position = self.initial_position.clone()
        self._time = 0.0
        
        device = self.initial_position.device
        dim = self.initial_position.shape[0]
        
        # 初始化默认值
        if self.velocity is None:
            self.velocity = torch.zeros(dim, device=device)
            
        if self.center is None:
            self.center = self.initial_position.clone()
            
        if self.direction is None:
            self.direction = torch.zeros(dim, device=device)
            self.direction[0] = 1.0  # 默认X方向
            
        self._current_velocity = self.velocity.clone()
        
    def get_position(self, t: float) -> torch.Tensor:
        """
        获取指定时间的位置
        
        Args:
            t: 时间
            
        Returns:
            position: 位置向量
        """
        if self.motion_type == MotionType.STATIC:
            return self.initial_position.clone()
            
        elif self.motion_type == MotionType.LINEAR:
            return self.initial_position + self.velocity * t
            
        elif self.motion_type == MotionType.CIRCULAR:
            angle = self.initial_angle + self.angular_velocity * t
            offset = torch.zeros_like(self.center)
            offset[0] = self.orbit_radius * torch.cos(torch.tensor(angle))
            offset[1] = self.orbit_radius * torch.sin(torch.tensor(angle))
            return self.center + offset
            
        elif self.motion_type == MotionType.SINUSOIDAL:
            offset = self.amplitude * torch.sin(
                torch.tensor(2 * np.pi * self.frequency * t)
            ) * self.direction
            return self.initial_position + offset
            
        elif self.motion_type == MotionType.RANDOM_WALK:
            # 注意：随机游走不是时间的确定函数
            # 这里返回预计算的位置或当前位置
            return self._current_position.clone()
            
        else:
            raise ValueError(f"Unknown motion type: {self.motion_type}")
    
    def get_velocity(self, t: float) -> torch.Tensor:
        """
        获取指定时间的速度
        
        Args:
            t: 时间
            
        Returns:
            velocity: 速度向量
        """
        if self.motion_type == MotionType.STATIC:
            return torch.zeros_like(self.initial_position)
            
        elif self.motion_type == MotionType.LINEAR:
            return self.velocity.clone()
            
        elif self.motion_type == MotionType.CIRCULAR:
            angle = self.initial_angle + self.angular_velocity * t
            vel = torch.zeros_like(self.center)
            vel[0] = -self.orbit_radius * self.angular_velocity * torch.sin(torch.tensor(angle))
            vel[1] = self.orbit_radius * self.angular_velocity * torch.cos(torch.tensor(angle))
            return vel
            
        elif self.motion_type == MotionType.SINUSOIDAL:
            vel = self.amplitude * 2 * np.pi * self.frequency * torch.cos(
                torch.tensor(2 * np.pi * self.frequency * t)
            ) * self.direction
            return vel
            
        elif self.motion_type == MotionType.RANDOM_WALK:
            return self._current_velocity.clone()
            
        else:
            raise ValueError(f"Unknown motion type: {self.motion_type}")
    
    def step(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        时间前进一步 (主要用于随机游走)
        
        Args:
            dt: 时间步长
            
        Returns:
            position, velocity: 新的位置和速度
        """
        self._time += dt
        
        if self.motion_type == MotionType.RANDOM_WALK:
            # 随机游走：添加高斯噪声
            noise = torch.randn_like(self._current_position) * self.noise_std * np.sqrt(dt)
            self._current_position = self._current_position + noise
            self._current_velocity = noise / dt
        else:
            self._current_position = self.get_position(self._time)
            self._current_velocity = self.get_velocity(self._time)
            
        return self._current_position, self._current_velocity
    
    def reset(self):
        """重置到初始状态"""
        self._current_position = self.initial_position.clone()
        self._current_velocity = self.velocity.clone() if self.velocity is not None else torch.zeros_like(self.initial_position)
        self._time = 0.0


class DynamicEnvironment2D:
    """
    2D动态环境
    
    在静态环境基础上添加动态障碍物支持。
    """
    
    def __init__(
        self,
        workspace_bounds: List[float] = [-1.0, 1.0, -1.0, 1.0],
        static_obstacles: Optional[List[Dict]] = None,
        dynamic_obstacles: Optional[List[DynamicObstacle]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            workspace_bounds: 工作空间边界 [x_min, x_max, y_min, y_max]
            static_obstacles: 静态障碍物列表 [{"position": [x,y], "radius": r}, ...]
            dynamic_obstacles: 动态障碍物列表
            device: 计算设备
        """
        self.workspace_bounds = workspace_bounds
        self.device = device
        
        # 静态障碍物
        self.static_obstacles = static_obstacles or []
        
        # 动态障碍物
        self.dynamic_obstacles = dynamic_obstacles or []
        
        # 当前时间
        self._time = 0.0
        
    def add_dynamic_obstacle(self, obstacle: DynamicObstacle):
        """添加动态障碍物"""
        self.dynamic_obstacles.append(obstacle)
        
    def add_random_dynamic_obstacles(
        self,
        num_obstacles: int = 3,
        radius_range: Tuple[float, float] = (0.08, 0.12),
        velocity_range: Tuple[float, float] = (0.1, 0.3),
        motion_types: Optional[List[MotionType]] = None,
    ):
        """
        添加随机动态障碍物
        
        Args:
            num_obstacles: 障碍物数量
            radius_range: 半径范围
            velocity_range: 速度范围
            motion_types: 允许的运动类型列表
        """
        if motion_types is None:
            motion_types = [MotionType.LINEAR, MotionType.CIRCULAR]
            
        x_min, x_max, y_min, y_max = self.workspace_bounds
        
        for i in range(num_obstacles):
            # 随机位置
            position = torch.tensor([
                np.random.uniform(x_min + 0.2, x_max - 0.2),
                np.random.uniform(y_min + 0.2, y_max - 0.2),
            ], device=self.device)
            
            # 随机半径
            radius = np.random.uniform(*radius_range)
            
            # 随机运动类型
            motion_type = np.random.choice(motion_types)
            
            # 随机速度
            speed = np.random.uniform(*velocity_range)
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = torch.tensor([
                speed * np.cos(angle),
                speed * np.sin(angle),
            ], device=self.device)
            
            obstacle = DynamicObstacle(
                obstacle_id=len(self.dynamic_obstacles),
                initial_position=position,
                radius=radius,
                motion_type=motion_type,
                velocity=velocity,
                angular_velocity=np.random.uniform(0.3, 0.8),
                orbit_radius=np.random.uniform(0.2, 0.4),
                center=position.clone(),
            )
            
            self.dynamic_obstacles.append(obstacle)
            
    def get_dynamic_obstacle_states(self, t: Optional[float] = None) -> List:
        """
        获取指定时间的所有动态障碍物状态
        
        Args:
            t: 时间 (None则使用当前时间)
            
        Returns:
            states: ObstacleState列表 (兼容obstacle_predictor)
        """
        from .obstacle_predictor import ObstacleState
        
        if t is None:
            t = self._time
            
        states = []
        for obs in self.dynamic_obstacles:
            state = ObstacleState(
                position=obs.get_position(t),
                velocity=obs.get_velocity(t),
                radius=obs.radius,
                obstacle_id=obs.obstacle_id,
                timestamp=t,
            )
            states.append(state)
        return states
    
    def step(self, dt: float):
        """
        环境时间前进一步
        
        Args:
            dt: 时间步长
        """
        self._time += dt
        for obs in self.dynamic_obstacles:
            obs.step(dt)
            
    def reset(self):
        """重置环境"""
        self._time = 0.0
        for obs in self.dynamic_obstacles:
            obs.reset()
            
    def get_all_obstacle_positions(self, t: Optional[float] = None) -> Tuple[List, List]:
        """
        获取所有障碍物位置 (静态+动态)
        
        Returns:
            positions: 位置列表
            radii: 半径列表
        """
        if t is None:
            t = self._time
            
        positions = []
        radii = []
        
        # 静态障碍物
        for obs in self.static_obstacles:
            positions.append(torch.tensor(obs["position"], device=self.device))
            radii.append(obs["radius"])
            
        # 动态障碍物
        for obs in self.dynamic_obstacles:
            positions.append(obs.get_position(t))
            radii.append(obs.radius)
            
        return positions, radii
    
    @property
    def current_time(self) -> float:
        return self._time
    
    @property
    def num_dynamic_obstacles(self) -> int:
        return len(self.dynamic_obstacles)
    
    def __repr__(self) -> str:
        return (
            f"DynamicEnvironment2D("
            f"bounds={self.workspace_bounds}, "
            f"n_static={len(self.static_obstacles)}, "
            f"n_dynamic={len(self.dynamic_obstacles)}, "
            f"time={self._time:.2f})"
        )


def create_simple_dynamic_env(
    num_dynamic: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DynamicEnvironment2D:
    """
    创建简单的动态测试环境
    
    Args:
        num_dynamic: 动态障碍物数量
        device: 计算设备
        
    Returns:
        env: 动态环境实例
    """
    # 静态障碍物 (类似EnvSimple2D)
    static_obstacles = [
        {"position": [-0.5, 0.3], "radius": 0.12},
        {"position": [0.3, 0.6], "radius": 0.12},
        {"position": [-0.5, -0.5], "radius": 0.12},
        {"position": [0.5, -0.5], "radius": 0.12},
    ]
    
    env = DynamicEnvironment2D(
        workspace_bounds=[-1.0, 1.0, -1.0, 1.0],
        static_obstacles=static_obstacles,
        device=device,
    )
    
    # 添加动态障碍物
    env.add_random_dynamic_obstacles(
        num_obstacles=num_dynamic,
        radius_range=(0.08, 0.12),
        velocity_range=(0.1, 0.25),
        motion_types=[MotionType.LINEAR, MotionType.CIRCULAR],
    )
    
    return env

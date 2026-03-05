"""
时空SDF (Spatio-Temporal Signed Distance Field)

将传统的3D SDF扩展到4D，增加时间维度以处理动态障碍物。
SDF(x, y, t) 表示在时间t，空间位置(x,y)到最近动态障碍物的距离。

Author: Dynamic MPD Project
Date: 2026-01-22
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ObstacleTrajectory:
    """动态障碍物轨迹数据结构"""
    positions: torch.Tensor  # [T, 2] 或 [T, 3] - 位置序列
    radii: float             # 障碍物半径
    timestamps: torch.Tensor # [T] - 时间戳
    obstacle_id: int = 0     # 障碍物ID
    
    @property
    def num_timesteps(self) -> int:
        return self.positions.shape[0]


class SpatioTemporalSDF:
    """
    时空签名距离场
    
    将动态障碍物的预测轨迹转换为4D (或3D for 2D场景) 距离场，
    用于运动规划中的时空碰撞检测和引导。
    
    Attributes:
        workspace_bounds: 工作空间边界 [x_min, x_max, y_min, y_max, (z_min, z_max)]
        spatial_resolution: 空间网格分辨率
        time_horizon: 时间范围
        num_time_steps: 时间步数
        sdf_grid: 时空SDF网格 [X, Y, T] 或 [X, Y, Z, T]
    """
    
    def __init__(
        self,
        workspace_bounds: List[float],
        spatial_resolution: float = 0.01,
        time_horizon: float = 10.0,
        num_time_steps: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化时空SDF
        
        Args:
            workspace_bounds: 工作空间边界 [x_min, x_max, y_min, y_max]
            spatial_resolution: 空间网格分辨率 (米)
            time_horizon: 预测时间范围 (秒)
            num_time_steps: 时间离散步数
            device: 计算设备
        """
        self.workspace_bounds = workspace_bounds
        self.spatial_resolution = spatial_resolution
        self.time_horizon = time_horizon
        self.num_time_steps = num_time_steps
        self.device = device
        
        # 计算网格尺寸
        self.dim = len(workspace_bounds) // 2  # 2D or 3D
        
        if self.dim == 2:
            x_min, x_max, y_min, y_max = workspace_bounds
            self.nx = int((x_max - x_min) / spatial_resolution)
            self.ny = int((y_max - y_min) / spatial_resolution)
            self.grid_shape = (self.nx, self.ny, num_time_steps)
        else:  # 3D
            x_min, x_max, y_min, y_max, z_min, z_max = workspace_bounds
            self.nx = int((x_max - x_min) / spatial_resolution)
            self.ny = int((y_max - y_min) / spatial_resolution)
            self.nz = int((z_max - z_min) / spatial_resolution)
            self.grid_shape = (self.nx, self.ny, self.nz, num_time_steps)
            
        # 时间分辨率
        self.dt = time_horizon / num_time_steps
        
        # 初始化SDF网格为无穷大(表示没有障碍物)
        self.sdf_grid = None
        self._initialize_grid()
        
        # 创建空间坐标网格
        self._create_coordinate_grids()
        
    def _initialize_grid(self):
        """初始化SDF网格"""
        self.sdf_grid = torch.full(
            self.grid_shape, 
            fill_value=float('inf'),
            device=self.device,
            dtype=torch.float32
        )
        
    def _create_coordinate_grids(self):
        """创建空间坐标网格用于SDF计算"""
        if self.dim == 2:
            x_min, x_max, y_min, y_max = self.workspace_bounds
            x = torch.linspace(x_min, x_max, self.nx, device=self.device)
            y = torch.linspace(y_min, y_max, self.ny, device=self.device)
            self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing='ij')
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = self.workspace_bounds
            x = torch.linspace(x_min, x_max, self.nx, device=self.device)
            y = torch.linspace(y_min, y_max, self.ny, device=self.device)
            z = torch.linspace(z_min, z_max, self.nz, device=self.device)
            self.grid_x, self.grid_y, self.grid_z = torch.meshgrid(x, y, z, indexing='ij')
            
    def compute_from_obstacle_trajectories(
        self,
        obstacle_trajectories: List[ObstacleTrajectory],
        static_sdf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从障碍物轨迹计算时空SDF
        
        Args:
            obstacle_trajectories: 动态障碍物轨迹列表
            static_sdf: 静态障碍物SDF (可选，会与动态SDF合并)
            
        Returns:
            sdf_grid: [X, Y, T] 或 [X, Y, Z, T] 时空SDF网格
        """
        # 重置网格
        self._initialize_grid()
        
        # 对每个时间步计算SDF
        for t_idx in range(self.num_time_steps):
            t = t_idx * self.dt  # 实际时间
            
            # 计算该时刻的动态障碍物SDF
            sdf_t = self._compute_sdf_at_time(obstacle_trajectories, t, t_idx)
            
            # 如果有静态SDF，取最小值
            if static_sdf is not None:
                sdf_t = torch.minimum(sdf_t, static_sdf)
                
            # 存入网格
            if self.dim == 2:
                self.sdf_grid[:, :, t_idx] = sdf_t
            else:
                self.sdf_grid[:, :, :, t_idx] = sdf_t
                
        return self.sdf_grid
    
    def _compute_sdf_at_time(
        self,
        obstacle_trajectories: List[ObstacleTrajectory],
        t: float,
        t_idx: int,
    ) -> torch.Tensor:
        """
        计算单个时间步的SDF
        
        Args:
            obstacle_trajectories: 障碍物轨迹列表
            t: 当前时间
            t_idx: 时间索引
            
        Returns:
            sdf: [X, Y] 或 [X, Y, Z] 该时刻的SDF
        """
        if self.dim == 2:
            sdf = torch.full((self.nx, self.ny), float('inf'), device=self.device)
        else:
            sdf = torch.full((self.nx, self.ny, self.nz), float('inf'), device=self.device)
            
        for obs_traj in obstacle_trajectories:
            # 获取障碍物在t时刻的位置 (插值)
            obs_pos = self._interpolate_position(obs_traj, t_idx)
            
            if obs_pos is None:
                continue
                
            # 计算到该障碍物的距离
            if self.dim == 2:
                dist = torch.sqrt(
                    (self.grid_x - obs_pos[0])**2 + 
                    (self.grid_y - obs_pos[1])**2
                ) - obs_traj.radii
            else:
                dist = torch.sqrt(
                    (self.grid_x - obs_pos[0])**2 + 
                    (self.grid_y - obs_pos[1])**2 +
                    (self.grid_z - obs_pos[2])**2
                ) - obs_traj.radii
                
            # 取最小值 (多个障碍物)
            sdf = torch.minimum(sdf, dist)
            
        return sdf
    
    def _interpolate_position(
        self, 
        obs_traj: ObstacleTrajectory, 
        t_idx: int
    ) -> Optional[torch.Tensor]:
        """
        插值获取障碍物在指定时间索引处的位置
        
        使用线性插值将SDF时间索引映射到障碍物轨迹时间
        
        Args:
            obs_traj: 障碍物轨迹
            t_idx: SDF的时间索引 [0, num_time_steps)
            
        Returns:
            position: 插值位置 [2] 或 [3]
        """
        # 将SDF时间索引映射到障碍物轨迹的浮点索引
        # t_idx: [0, num_time_steps) -> obs_idx: [0, num_timesteps)
        obs_num_steps = obs_traj.num_timesteps
        
        # 线性映射
        obs_idx_float = t_idx * (obs_num_steps - 1) / max(self.num_time_steps - 1, 1)
        
        if obs_idx_float >= obs_num_steps - 1:
            return obs_traj.positions[-1]
        elif obs_idx_float <= 0:
            return obs_traj.positions[0]
        else:
            # 线性插值
            idx_low = int(obs_idx_float)
            idx_high = idx_low + 1
            alpha = obs_idx_float - idx_low
            
            pos_low = obs_traj.positions[idx_low]
            pos_high = obs_traj.positions[idx_high]
            
            return pos_low * (1 - alpha) + pos_high * alpha
    
    def query_trajectory(
        self,
        trajectory: torch.Tensor,
        trajectory_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        查询轨迹上每个点的SDF值
        
        Args:
            trajectory: [T, 2] 或 [T, 3] 机器人轨迹
            trajectory_times: [T] 轨迹时间戳 (可选，默认均匀分布)
            
        Returns:
            sdf_values: [T] 轨迹上每个点的SDF值
        """
        T = trajectory.shape[0]
        
        if trajectory_times is None:
            # 假设轨迹时间均匀分布在整个时间范围
            trajectory_times = torch.linspace(0, self.time_horizon, T, device=self.device)
            
        sdf_values = torch.zeros(T, device=self.device)
        
        for i in range(T):
            pos = trajectory[i]
            t = trajectory_times[i]
            sdf_values[i] = self.query_point(pos, t)
            
        return sdf_values
    
    def query_point(
        self,
        position: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        """
        查询单个时空点的SDF值 (三线性插值)
        
        Args:
            position: [2] 或 [3] 空间位置
            time: 时间
            
        Returns:
            sdf_value: 标量SDF值
        """
        # 转换为网格坐标
        if self.dim == 2:
            x_min, x_max, y_min, y_max = self.workspace_bounds
            
            # 归一化坐标到 [0, 1]
            x_norm = (position[0] - x_min) / (x_max - x_min)
            y_norm = (position[1] - y_min) / (y_max - y_min)
            t_norm = time / self.time_horizon
            
            # 转换为网格索引 (浮点)
            x_idx = x_norm * (self.nx - 1)
            y_idx = y_norm * (self.ny - 1)
            t_idx = t_norm * (self.num_time_steps - 1)
            
            # 三线性插值
            return self._trilinear_interpolate_2d(x_idx, y_idx, t_idx)
        else:
            # 3D场景
            x_min, x_max, y_min, y_max, z_min, z_max = self.workspace_bounds
            
            x_norm = (position[0] - x_min) / (x_max - x_min)
            y_norm = (position[1] - y_min) / (y_max - y_min)
            z_norm = (position[2] - z_min) / (z_max - z_min)
            t_norm = time / self.time_horizon
            
            x_idx = x_norm * (self.nx - 1)
            y_idx = y_norm * (self.ny - 1)
            z_idx = z_norm * (self.nz - 1)
            t_idx = t_norm * (self.num_time_steps - 1)
            
            return self._quadrilinear_interpolate_3d(x_idx, y_idx, z_idx, t_idx)
    
    def _trilinear_interpolate_2d(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        2D空间 + 时间的三线性插值
        """
        # 确保输入是tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
            
        # 边界裁剪
        x = torch.clamp(x, 0.0, float(self.nx - 1.001))
        y = torch.clamp(y, 0.0, float(self.ny - 1.001))
        t = torch.clamp(t, 0.0, float(self.num_time_steps - 1.001))
        
        # 整数和小数部分
        x0, y0, t0 = int(x.item()), int(y.item()), int(t.item())
        x1, y1, t1 = min(x0 + 1, self.nx - 1), min(y0 + 1, self.ny - 1), min(t0 + 1, self.num_time_steps - 1)
        
        xd = x - x0
        yd = y - y0
        td = t - t0
        
        # 8个角点的值
        c000 = self.sdf_grid[x0, y0, t0]
        c001 = self.sdf_grid[x0, y0, t1]
        c010 = self.sdf_grid[x0, y1, t0]
        c011 = self.sdf_grid[x0, y1, t1]
        c100 = self.sdf_grid[x1, y0, t0]
        c101 = self.sdf_grid[x1, y0, t1]
        c110 = self.sdf_grid[x1, y1, t0]
        c111 = self.sdf_grid[x1, y1, t1]
        
        # 三线性插值
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        
        c = c0 * (1 - td) + c1 * td
        
        return c
    
    def _quadrilinear_interpolate_3d(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        3D空间 + 时间的四线性插值
        """
        # 简化实现：先做空间三线性插值，再做时间线性插值
        x = torch.clamp(x, 0, self.nx - 1.001)
        y = torch.clamp(y, 0, self.ny - 1.001)
        z = torch.clamp(z, 0, self.nz - 1.001)
        t = torch.clamp(t, 0, self.num_time_steps - 1.001)
        
        t0 = int(t)
        t1 = min(t0 + 1, self.num_time_steps - 1)
        td = t - t0
        
        # 两个时间步的空间插值
        v0 = self._spatial_trilinear_3d(x, y, z, t0)
        v1 = self._spatial_trilinear_3d(x, y, z, t1)
        
        return v0 * (1 - td) + v1 * td
    
    def _spatial_trilinear_3d(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """3D空间三线性插值"""
        x0, y0, z0 = int(x), int(y), int(z)
        x1 = min(x0 + 1, self.nx - 1)
        y1 = min(y0 + 1, self.ny - 1)
        z1 = min(z0 + 1, self.nz - 1)
        
        xd, yd, zd = x - x0, y - y0, z - z0
        
        c000 = self.sdf_grid[x0, y0, z0, t_idx]
        c001 = self.sdf_grid[x0, y0, z1, t_idx]
        c010 = self.sdf_grid[x0, y1, z0, t_idx]
        c011 = self.sdf_grid[x0, y1, z1, t_idx]
        c100 = self.sdf_grid[x1, y0, z0, t_idx]
        c101 = self.sdf_grid[x1, y0, z1, t_idx]
        c110 = self.sdf_grid[x1, y1, z0, t_idx]
        c111 = self.sdf_grid[x1, y1, z1, t_idx]
        
        c00 = c000 * (1 - zd) + c001 * zd
        c01 = c010 * (1 - zd) + c011 * zd
        c10 = c100 * (1 - zd) + c101 * zd
        c11 = c110 * (1 - zd) + c111 * zd
        
        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd
        
        return c0 * (1 - xd) + c1 * xd
    
    def compute_gradient(
        self,
        position: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        """
        计算SDF在指定时空点的梯度 (用于引导)
        
        Args:
            position: [2] 或 [3] 空间位置
            time: 时间
            
        Returns:
            gradient: [2] 或 [3] 空间梯度 (不包括时间梯度)
        """
        eps = self.spatial_resolution * 0.5
        
        if self.dim == 2:
            grad = torch.zeros(2, device=self.device)
            
            # 中心差分
            grad[0] = (
                self.query_point(position + torch.tensor([eps, 0], device=self.device), time) -
                self.query_point(position - torch.tensor([eps, 0], device=self.device), time)
            ) / (2 * eps)
            
            grad[1] = (
                self.query_point(position + torch.tensor([0, eps], device=self.device), time) -
                self.query_point(position - torch.tensor([0, eps], device=self.device), time)
            ) / (2 * eps)
        else:
            grad = torch.zeros(3, device=self.device)
            
            grad[0] = (
                self.query_point(position + torch.tensor([eps, 0, 0], device=self.device), time) -
                self.query_point(position - torch.tensor([eps, 0, 0], device=self.device), time)
            ) / (2 * eps)
            
            grad[1] = (
                self.query_point(position + torch.tensor([0, eps, 0], device=self.device), time) -
                self.query_point(position - torch.tensor([0, eps, 0], device=self.device), time)
            ) / (2 * eps)
            
            grad[2] = (
                self.query_point(position + torch.tensor([0, 0, eps], device=self.device), time) -
                self.query_point(position - torch.tensor([0, 0, eps], device=self.device), time)
            ) / (2 * eps)
            
        return grad
    
    def get_time_slice(self, t_idx: int) -> torch.Tensor:
        """
        获取指定时间步的2D/3D SDF切片
        
        Args:
            t_idx: 时间索引
            
        Returns:
            sdf_slice: [X, Y] 或 [X, Y, Z]
        """
        if self.dim == 2:
            return self.sdf_grid[:, :, t_idx]
        else:
            return self.sdf_grid[:, :, :, t_idx]
    
    def __repr__(self) -> str:
        return (
            f"SpatioTemporalSDF("
            f"dim={self.dim}, "
            f"grid_shape={self.grid_shape}, "
            f"resolution={self.spatial_resolution}, "
            f"time_horizon={self.time_horizon}s, "
            f"device={self.device})"
        )

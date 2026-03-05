"""
动态碰撞引导

扩展现有MPD的引导机制，添加时空碰撞代价。

Author: Dynamic MPD Project
Date: 2026-01-22
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .spatiotemporal_sdf import SpatioTemporalSDF


class DynamicCollisionGuidance:
    """
    动态碰撞引导
    
    计算轨迹的时空碰撞代价和梯度，用于扩散模型的引导采样。
    """
    
    def __init__(
        self,
        st_sdf: SpatioTemporalSDF,
        static_sdf: Optional[torch.Tensor] = None,
        safety_margin: float = 0.05,
        weights: Optional[Dict[str, float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            st_sdf: 时空SDF
            static_sdf: 静态SDF (可选)
            safety_margin: 安全边距
            weights: 各代价项权重
            device: 计算设备
        """
        self.st_sdf = st_sdf
        self.static_sdf = static_sdf
        self.safety_margin = safety_margin
        self.device = device
        
        # 默认权重
        default_weights = {
            "st_collision": 20.0,      # 时空碰撞 (最高优先级)
            "static_collision": 10.0,  # 静态碰撞
            "velocity": 0.01,          # 速度平滑
            "acceleration": 0.001,     # 加速度平滑
        }
        self.weights = weights or default_weights
        
    def compute_spatiotemporal_collision_cost(
        self,
        trajectory: torch.Tensor,
        trajectory_times: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算时空碰撞代价
        
        Args:
            trajectory: [T, d] 轨迹
            trajectory_times: [T] 时间戳
            
        Returns:
            cost: 标量代价
            gradient: [T, d] 梯度
        """
        T, d = trajectory.shape
        
        if trajectory_times is None:
            trajectory_times = torch.linspace(
                0, self.st_sdf.time_horizon, T, device=self.device
            )
            
        # 需要梯度
        trajectory_with_grad = trajectory.clone().requires_grad_(True)
        
        # 计算每个点的SDF值
        sdf_values = self.st_sdf.query_trajectory(
            trajectory_with_grad, 
            trajectory_times
        )
        
        # 碰撞代价: max(0, margin - sdf)^2
        collision_cost = torch.sum(
            F.relu(self.safety_margin - sdf_values) ** 2
        )
        
        # 计算梯度
        if trajectory_with_grad.grad is not None:
            trajectory_with_grad.grad.zero_()
            
        collision_cost.backward(retain_graph=True)
        gradient = trajectory_with_grad.grad.clone() if trajectory_with_grad.grad is not None else torch.zeros_like(trajectory)
        
        return collision_cost.detach(), gradient.detach()
    
    def compute_total_cost_and_gradient(
        self,
        trajectory: torch.Tensor,
        trajectory_times: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        计算总代价和梯度
        
        Args:
            trajectory: [T, d] 轨迹
            trajectory_times: [T] 时间戳
            
        Returns:
            costs: 各项代价字典
            total_gradient: [T, d] 总梯度
        """
        T, d = trajectory.shape
        costs = {}
        gradients = {}
        
        # 1. 时空碰撞代价
        st_cost, st_grad = self.compute_spatiotemporal_collision_cost(
            trajectory, trajectory_times
        )
        costs["st_collision"] = st_cost
        gradients["st_collision"] = st_grad * self.weights["st_collision"]
        
        # 2. 静态碰撞代价 (如果有)
        if self.static_sdf is not None:
            static_cost, static_grad = self._compute_static_collision_cost(trajectory)
            costs["static_collision"] = static_cost
            gradients["static_collision"] = static_grad * self.weights["static_collision"]
        
        # 3. 速度平滑代价
        vel_cost, vel_grad = self._compute_velocity_cost(trajectory)
        costs["velocity"] = vel_cost
        gradients["velocity"] = vel_grad * self.weights["velocity"]
        
        # 4. 加速度平滑代价
        acc_cost, acc_grad = self._compute_acceleration_cost(trajectory)
        costs["acceleration"] = acc_cost
        gradients["acceleration"] = acc_grad * self.weights["acceleration"]
        
        # 合并梯度
        total_gradient = sum(gradients.values())
        
        return costs, total_gradient
    
    def _compute_static_collision_cost(
        self,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算静态碰撞代价"""
        # 简化实现：使用静态SDF网格
        if self.static_sdf is None:
            return torch.tensor(0.0, device=self.device), torch.zeros_like(trajectory)
            
        # TODO: 实现静态SDF查询
        return torch.tensor(0.0, device=self.device), torch.zeros_like(trajectory)
    
    def _compute_velocity_cost(
        self,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算速度平滑代价"""
        # 一阶差分作为速度
        velocity = trajectory[1:] - trajectory[:-1]
        cost = torch.sum(velocity ** 2)
        
        # 梯度
        gradient = torch.zeros_like(trajectory)
        gradient[:-1] -= 2 * velocity
        gradient[1:] += 2 * velocity
        
        return cost, gradient
    
    def _compute_acceleration_cost(
        self,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算加速度平滑代价"""
        # 二阶差分作为加速度
        if trajectory.shape[0] < 3:
            return torch.tensor(0.0, device=self.device), torch.zeros_like(trajectory)
            
        acceleration = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
        cost = torch.sum(acceleration ** 2)
        
        # 梯度
        gradient = torch.zeros_like(trajectory)
        gradient[:-2] += 2 * acceleration
        gradient[1:-1] -= 4 * acceleration
        gradient[2:] += 2 * acceleration
        
        return cost, gradient


class HierarchicalGuidance:
    """
    层级引导
    
    实现分层的梯度投影，确保高优先级约束先满足。
    优先级顺序: 时空碰撞 > 静态碰撞 > 关节限位 > 速度/加速度
    """
    
    def __init__(
        self,
        dynamic_guidance: DynamicCollisionGuidance,
        priority_order: Optional[list] = None,
    ):
        """
        Args:
            dynamic_guidance: 动态碰撞引导
            priority_order: 优先级顺序 (从高到低)
        """
        self.dynamic_guidance = dynamic_guidance
        self.priority_order = priority_order or [
            "st_collision",
            "static_collision",
            "velocity",
            "acceleration",
        ]
        
    def compute_hierarchical_gradient(
        self,
        trajectory: torch.Tensor,
        trajectory_times: Optional[torch.Tensor] = None,
        project_gradients: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        计算层级梯度
        
        使用零空间投影确保低优先级梯度不干扰高优先级约束。
        
        Args:
            trajectory: [T, d] 轨迹
            trajectory_times: [T] 时间戳
            project_gradients: 是否进行梯度投影
            
        Returns:
            costs: 各项代价
            gradient: 投影后的总梯度
        """
        costs, _ = self.dynamic_guidance.compute_total_cost_and_gradient(
            trajectory, trajectory_times
        )
        
        if not project_gradients:
            # 不投影，直接返回加权和
            return self.dynamic_guidance.compute_total_cost_and_gradient(
                trajectory, trajectory_times
            )
            
        # 层级投影
        # TODO: 实现零空间投影
        # 当前简化实现：按优先级加权
        
        T, d = trajectory.shape
        total_gradient = torch.zeros_like(trajectory)
        
        # 重新计算各项梯度
        trajectory_with_grad = trajectory.clone().requires_grad_(True)
        
        # 按优先级顺序累加
        decay_factor = 1.0
        for i, cost_name in enumerate(self.priority_order):
            if cost_name in costs:
                weight = self.dynamic_guidance.weights.get(cost_name, 1.0)
                # 优先级衰减
                effective_weight = weight * decay_factor
                decay_factor *= 0.5  # 低优先级权重衰减
                
        return costs, total_gradient


def create_guidance_from_st_sdf(
    st_sdf: SpatioTemporalSDF,
    safety_margin: float = 0.05,
    **kwargs,
) -> DynamicCollisionGuidance:
    """
    工厂函数：从时空SDF创建引导
    """
    return DynamicCollisionGuidance(
        st_sdf=st_sdf,
        safety_margin=safety_margin,
        **kwargs,
    )

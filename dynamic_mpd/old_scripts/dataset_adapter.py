"""
MPD Dataset适配器

创建轻量级的dataset接口，用于直接调用预训练扩散模型生成轨迹。

Author: Dynamic MPD Project
Date: 2026-01-23
"""

import os
import sys
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import numpy as np

# 确保MPD路径正确
MPD_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, MPD_ROOT)
sys.path.insert(0, os.path.join(MPD_ROOT, 'mpd', 'torch_robotics'))


@dataclass
class DatasetConfig:
    """Dataset配置"""
    state_dim: int = 2  # 状态维度 (2D: x, y)
    n_learnable_control_points: int = 16  # 可学习控制点数 (训练时的horizon)
    context_qs: bool = True  # 使用起点/终点作为上下文
    context_ee_goal_pose: bool = False  # 不使用末端执行器目标
    zero_vel_at_boundary: bool = True  # 边界零速度约束
    zero_acc_at_boundary: bool = True  # 边界零加速度约束
    # 归一化范围
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    

class LightweightDatasetAdapter:
    """
    轻量级Dataset适配器
    
    提供必要的接口用于调用预训练MPD模型，无需加载完整数据集。
    模型期望:
        - hard_conds: Dict[int, Tensor] 硬约束条件
        - context_d: Dict[str, Tensor] 上下文字典
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        normalizer_stats: Optional[Dict] = None,
        tensor_args: Optional[Dict] = None,
    ):
        """
        Args:
            config: Dataset配置
            normalizer_stats: 归一化统计量 (均值、标准差等)
            tensor_args: PyTorch tensor参数
        """
        self.config = config
        self.device = tensor_args["device"] if tensor_args else "cuda"
        self.tensor_args = tensor_args or {"device": self.device, "dtype": torch.float32}
        
        # Dataset属性
        self.state_dim = config.state_dim
        self.n_learnable_control_points = config.n_learnable_control_points
        self.control_points_dim = (config.n_learnable_control_points, config.state_dim)
        self.context_qs = config.context_qs
        self.context_ee_goal_pose = config.context_ee_goal_pose
        
        # 字段名称 (与MPD兼容)
        self.field_key_control_points = "control_points"
        self.field_key_q_start = "q_start"
        self.field_key_q_goal = "q_goal"
        self.field_key_context_qs = "qs"
        
        # 归一化参数
        self._setup_normalizer(normalizer_stats)
        
    def _setup_normalizer(self, stats: Optional[Dict] = None):
        """设置归一化参数"""
        if stats is not None:
            self.normalizer_stats = stats
        else:
            # 使用配置中的范围
            cfg = self.config
            # 控制点归一化: [x_min, x_max, y_min, y_max] -> [-1, 1]
            self.normalizer_stats = {
                self.field_key_control_points: {
                    "min": torch.tensor([cfg.x_min, cfg.y_min], **self.tensor_args),
                    "max": torch.tensor([cfg.x_max, cfg.y_max], **self.tensor_args),
                },
                self.field_key_context_qs: {
                    "min": torch.tensor([cfg.x_min, cfg.y_min, cfg.x_min, cfg.y_min], **self.tensor_args),
                    "max": torch.tensor([cfg.x_max, cfg.y_max, cfg.x_max, cfg.y_max], **self.tensor_args),
                },
            }
            
    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        归一化数据到 [-1, 1] 范围
        """
        if key not in self.normalizer_stats:
            return x
            
        stats = self.normalizer_stats[key]
        x_min = stats["min"].to(x.device)
        x_max = stats["max"].to(x.device)
        
        # 广播处理
        while x_min.dim() < x.dim():
            x_min = x_min.unsqueeze(0)
            x_max = x_max.unsqueeze(0)
        
        # 线性归一化到 [-1, 1]
        x_normalized = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        return x_normalized
        
    def unnormalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        反归一化
        """
        if key not in self.normalizer_stats:
            return x
            
        stats = self.normalizer_stats[key]
        x_min = stats["min"].to(x.device)
        x_max = stats["max"].to(x.device)
        
        # 广播处理
        while x_min.dim() < x.dim():
            x_min = x_min.unsqueeze(0)
            x_max = x_max.unsqueeze(0)
        
        x_unnormalized = (x + 1) / 2 * (x_max - x_min) + x_min
        return x_unnormalized
        
    def normalize_control_points(self, x: torch.Tensor) -> torch.Tensor:
        """归一化控制点"""
        return self.normalize(x, self.field_key_control_points)
        
    def unnormalize_control_points(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化控制点"""
        return self.unnormalize(x, self.field_key_control_points)
        
    def normalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """归一化轨迹"""
        return self.normalize_control_points(traj)
        
    def unnormalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """反归一化轨迹"""
        return self.unnormalize_control_points(traj)
        
    def create_data_sample_normalized(
        self,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
        ee_pose_goal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        创建归一化的数据样本 (MPD兼容格式)
        
        Args:
            q_start: 起点位置 [d]
            q_goal: 终点位置 [d]
            ee_pose_goal: 末端执行器目标 (可选)
            
        Returns:
            包含hard_conds和context的数据字典
        """
        data_sample = {}
        
        # 确保是1D tensor
        if q_start.dim() > 1:
            q_start = q_start.squeeze()
        if q_goal.dim() > 1:
            q_goal = q_goal.squeeze()
        
        # 构建起点/终点上下文 (qs)
        qs = torch.cat([q_start, q_goal], dim=-1)  # [2*d]
        data_sample[self.field_key_context_qs] = qs
        data_sample[f"{self.field_key_context_qs}_normalized"] = self.normalize(
            qs, self.field_key_context_qs
        )
        
        # 起点/终点位置
        data_sample[self.field_key_q_start] = q_start
        data_sample[self.field_key_q_goal] = q_goal
        
        # 归一化的起点/终点 (作为控制点边界)
        q_start_normalized = self.normalize_control_points(q_start)
        q_goal_normalized = self.normalize_control_points(q_goal)
        
        # 保存归一化版本
        data_sample[f"{self.field_key_q_start}_normalized"] = q_start_normalized
        data_sample[f"{self.field_key_q_goal}_normalized"] = q_goal_normalized
        
        # 构建hard conditions (边界条件)
        hard_conds = self.get_hard_conditions(q_start_normalized, q_goal_normalized)
        data_sample["hard_conds"] = hard_conds
        
        return data_sample
        
    def get_hard_conditions(
        self,
        q_start_normalized: torch.Tensor,
        q_goal_normalized: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        获取硬约束条件
        
        对于B-spline轨迹，起点和终点的控制点是固定的。
        MPD模型使用这些约束来固定轨迹的边界点。
        
        Args:
            q_start_normalized: 归一化的起点
            q_goal_normalized: 归一化的终点
            
        Returns:
            hard_conds: {index: value} 字典
        """
        horizon = self.n_learnable_control_points
        cfg = self.config
        
        hard_conds = {}
        
        if not self.context_qs:
            # 如果不使用qs作为上下文，则将起点/终点作为硬约束
            hard_conds[0] = q_start_normalized
            if cfg.zero_vel_at_boundary:
                hard_conds[1] = q_start_normalized  # 零速度约束
            if cfg.zero_acc_at_boundary:
                hard_conds[2] = q_start_normalized  # 零加速度约束
            
            if not self.context_ee_goal_pose:
                hard_conds[horizon - 1] = q_goal_normalized
                if cfg.zero_vel_at_boundary:
                    hard_conds[horizon - 2] = q_goal_normalized
                if cfg.zero_acc_at_boundary:
                    hard_conds[horizon - 3] = q_goal_normalized
        
        return hard_conds
        
    def build_context(self, data_sample: Dict) -> Dict[str, torch.Tensor]:
        """
        构建上下文字典 (MPD兼容格式)
        
        Args:
            data_sample: 数据样本 (来自create_data_sample_normalized)
            
        Returns:
            context_d: 上下文字典
        """
        context_d = {}
        
        if self.context_qs:
            context_d = {
                self.field_key_context_qs: data_sample[self.field_key_context_qs],
                f"{self.field_key_context_qs}_normalized": data_sample[
                    f"{self.field_key_context_qs}_normalized"
                ],
            }
            
        return context_d
    
    def prepare_inference_inputs(
        self,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        准备模型推理的输入 (hard_conds 和 context_d)
        
        这是调用model.run_inference()的便捷方法。
        
        Args:
            q_start: 起点位置 [d]
            q_goal: 终点位置 [d]
            
        Returns:
            hard_conds: 硬约束字典
            context_d: 上下文字典
        """
        # 创建数据样本
        data_sample = self.create_data_sample_normalized(q_start, q_goal)
        
        # 提取hard_conds和context
        hard_conds = data_sample["hard_conds"]
        context_d = self.build_context(data_sample)
        
        return hard_conds, context_d


class MPDDatasetAdapter(LightweightDatasetAdapter):
    """
    MPD Dataset适配器
    
    从训练参数自动加载归一化统计量。
    """
    
    def __init__(
        self,
        training_args: Dict,
        tensor_args: Optional[Dict] = None,
    ):
        """
        Args:
            training_args: 从args.yaml加载的训练参数
            tensor_args: PyTorch参数
        """
        # 从训练参数提取配置
        bspline_points = training_args.get("bspline_num_control_points_desired", 18)
        
        config = DatasetConfig(
            state_dim=training_args.get("state_dim", 2),
            n_learnable_control_points=bspline_points - 2,  # 减去起点和终点
            context_qs=training_args.get("context_qs", True),
            context_ee_goal_pose=training_args.get("context_ee_goal_pose", False),
        )
        
        super().__init__(config, tensor_args=tensor_args)
        
        self.training_args = training_args
        
        # 尝试从训练参数获取归一化统计量
        self._load_normalizer_from_args(training_args)
        
    @classmethod
    def from_training_args(cls, args, tensor_args: Optional[Dict] = None):
        """
        从训练参数创建适配器
        
        Args:
            args: DotMap或Dict类型的训练参数
            tensor_args: PyTorch参数
            
        Returns:
            MPDDatasetAdapter实例
        """
        # 转换DotMap为dict
        if hasattr(args, 'toDict'):
            args_dict = args.toDict()
        elif hasattr(args, '__dict__'):
            args_dict = dict(args)
        else:
            args_dict = args
            
        return cls(args_dict, tensor_args)
        
    def _load_normalizer_from_args(self, args: Dict):
        """从训练参数加载归一化器"""
        # 尝试多种方式获取workspace bounds
        bounds = None
        
        if "workspace_bounds" in args:
            bounds = args["workspace_bounds"]
        elif "env_config" in args:
            env_config = args["env_config"]
            if isinstance(env_config, dict) and "workspace_bounds" in env_config:
                bounds = env_config["workspace_bounds"]
                
        if bounds is not None:
            # bounds 格式: [x_min, x_max, y_min, y_max, ...]
            d = self.state_dim
            mins = torch.tensor([bounds[2*i] for i in range(d)], **self.tensor_args)
            maxs = torch.tensor([bounds[2*i+1] for i in range(d)], **self.tensor_args)
            
            self.normalizer_stats[self.field_key_control_points] = {
                "min": mins,
                "max": maxs,
            }
            
            # context_qs 是 [start, goal] 拼接
            self.normalizer_stats[self.field_key_context_qs] = {
                "min": torch.cat([mins, mins]),
                "max": torch.cat([maxs, maxs]),
            }
            print(f"[INFO] 从训练参数加载归一化范围: x=[{mins[0]:.2f}, {maxs[0]:.2f}], y=[{mins[1]:.2f}, {maxs[1]:.2f}]")
        else:
            print(f"[INFO] 使用默认归一化范围: [-1, 1]")


def create_simple_2d_adapter(
    workspace_bounds: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    n_control_points: int = 18,
    device: str = "cuda",
) -> LightweightDatasetAdapter:
    """
    创建简单的2D环境适配器
    
    Args:
        workspace_bounds: [x_min, x_max, y_min, y_max]
        n_control_points: 控制点数量
        device: 计算设备
        
    Returns:
        配置好的适配器
    """
    tensor_args = {"device": device, "dtype": torch.float32}
    
    config = DatasetConfig(
        state_dim=2,
        n_learnable_control_points=n_control_points - 2,
        context_qs=True,
        context_ee_goal_pose=False,
    )
    
    x_min, x_max, y_min, y_max = workspace_bounds
    
    normalizer_stats = {
        "control_points": {
            "min": torch.tensor([x_min, y_min], **tensor_args),
            "max": torch.tensor([x_max, y_max], **tensor_args),
        },
        "qs": {
            "min": torch.tensor([x_min, y_min, x_min, y_min], **tensor_args),
            "max": torch.tensor([x_max, y_max, x_max, y_max], **tensor_args),
        },
    }
    
    adapter = LightweightDatasetAdapter(config, normalizer_stats, tensor_args)
    return adapter


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Dataset适配器测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建适配器
    adapter = create_simple_2d_adapter(
        workspace_bounds=(-1.0, 1.0, -1.0, 1.0),
        n_control_points=18,
        device=device,
    )
    
    print(f"\n配置:")
    print(f"  状态维度: {adapter.state_dim}")
    print(f"  控制点数: {adapter.n_learnable_control_points}")
    print(f"  使用qs上下文: {adapter.context_qs}")
    
    # 测试数据样本创建
    start = torch.tensor([-0.8, -0.8], device=device)
    goal = torch.tensor([0.8, 0.8], device=device)
    
    data_sample = adapter.create_data_sample_normalized(start, goal)
    
    print(f"\n数据样本:")
    for key, value in data_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"  {key}: {len(value)} entries")
            
    # 测试上下文
    context = adapter.build_context(data_sample)
    print(f"\n上下文:")
    for key, value in context.items():
        print(f"  {key}: shape={value.shape}")
        
    # 测试归一化
    test_point = torch.tensor([0.5, 0.5], device=device)
    normalized = adapter.normalize_control_points(test_point)
    unnormalized = adapter.unnormalize_control_points(normalized)
    
    print(f"\n归一化测试:")
    print(f"  原始: {test_point.tolist()}")
    print(f"  归一化: {normalized.tolist()}")
    print(f"  反归一化: {unnormalized.tolist()}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

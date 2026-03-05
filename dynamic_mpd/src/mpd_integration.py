"""
MPD模型集成模块

将预训练的MPD模型与动态环境集成，支持时空SDF引导。

Author: Dynamic MPD Project
Date: 2026-01-23
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# 注意: 需要在torch之前导入isaacgym
try:
    import isaacgym
except ImportError:
    pass

import torch
import numpy as np
import einops
from dotmap import DotMap

# 确保MPD路径正确
MPD_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, MPD_ROOT)
sys.path.insert(0, os.path.join(MPD_ROOT, 'mpd', 'torch_robotics'))

MPD_AVAILABLE = False
DATASET_ADAPTER_AVAILABLE = False

try:
    from mpd.utils.loaders import load_params_from_yaml
    from mpd.models.diffusion_models import GaussianDiffusionModel
    from mpd.models.cvae.cvae import CVAEModel
    from torch_robotics.torch_utils.torch_utils import (
        freeze_torch_model_params, 
        dict_to_device,
        to_torch, 
        to_numpy,
    )
    MPD_AVAILABLE = True
    print("[INFO] MPD模块导入成功")
except ImportError as e:
    print(f"[WARN] MPD模块导入失败: {e}")

# 导入数据集适配器
try:
    from .dataset_adapter import (
        LightweightDatasetAdapter,
        MPDDatasetAdapter,
        DatasetConfig,
        create_simple_2d_adapter,
    )
    DATASET_ADAPTER_AVAILABLE = True
    print("[INFO] Dataset适配器导入成功")
except ImportError as e:
    print(f"[WARN] Dataset适配器导入失败: {e}")


@dataclass 
class MPDModelConfig:
    """MPD模型配置"""
    model_dir: str  # 模型目录
    model_selection: str = "bspline"  # bspline or waypoints
    use_ema: bool = True  # 是否使用EMA模型
    device: str = "cuda"
    
    # 扩散采样参数
    diffusion_sampling_method: str = "ddpm"
    n_diffusion_steps: int = 50
    t_start_guide_steps_fraction: float = 0.25
    
    # 轨迹参数
    n_trajectory_samples: int = 10
    num_T_pts: int = 64  # 轨迹时间点数


class PretrainedMPDLoader:
    """
    预训练MPD模型加载器
    
    负责加载预训练的扩散模型并提供推理接口。
    """
    
    def __init__(
        self,
        config: MPDModelConfig,
        tensor_args: Optional[Dict] = None,
    ):
        """
        Args:
            config: MPD模型配置
            tensor_args: PyTorch tensor参数
        """
        self.config = config
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.tensor_args = tensor_args or {"device": self.device, "dtype": torch.float32}
        
        self.model = None
        self.args_train = None
        self.dataset_adapter = None  # 数据集适配器
        self.is_loaded = False
        
    def load(self) -> bool:
        """
        加载预训练模型
        
        Returns:
            是否加载成功
        """
        if not MPD_AVAILABLE:
            print("[ERROR] MPD模块不可用，无法加载模型")
            return False
            
        model_dir = os.path.expandvars(self.config.model_dir)
        
        if not os.path.exists(model_dir):
            print(f"[ERROR] 模型目录不存在: {model_dir}")
            return False
            
        # 加载训练参数
        args_path = os.path.join(model_dir, "args.yaml")
        if not os.path.exists(args_path):
            print(f"[ERROR] 训练参数文件不存在: {args_path}")
            return False
            
        try:
            self.args_train = DotMap(load_params_from_yaml(args_path))
            print(f"[INFO] 加载训练参数: {args_path}")
        except Exception as e:
            print(f"[ERROR] 加载训练参数失败: {e}")
            return False
            
        # 加载模型权重
        model_filename = f'{"ema_" if self.config.use_ema else ""}model_current.pth'
        model_path = os.path.join(model_dir, "checkpoints", model_filename)
        
        if not os.path.exists(model_path):
            print(f"[ERROR] 模型文件不存在: {model_path}")
            return False
            
        try:
            print(f"[INFO] 加载模型: {model_path}")
            self.model = torch.load(model_path, map_location=self.tensor_args["device"])
            self.model.eval()
            freeze_torch_model_params(self.model)
            self.is_loaded = True
            print(f"[INFO] 模型加载成功!")
            print(f"       模型类型: {type(self.model).__name__}")
            if hasattr(self.model, 'n_diffusion_steps'):
                print(f"       扩散步数: {self.model.n_diffusion_steps}")
                
            # 创建数据集适配器
            if DATASET_ADAPTER_AVAILABLE:
                try:
                    self.dataset_adapter = MPDDatasetAdapter.from_training_args(
                        self.args_train, 
                        self.tensor_args
                    )
                    print(f"[INFO] 数据集适配器创建成功")
                except Exception as e:
                    print(f"[WARN] 创建数据集适配器失败: {e}")
                    # 回退到轻量级适配器
                    self.dataset_adapter = create_simple_2d_adapter(self.tensor_args)
                    print(f"[INFO] 使用默认轻量级适配器")
                    
        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            return False
            
        return True
        
    def get_model(self):
        """获取加载的模型"""
        return self.model
        
    def get_training_args(self):
        """获取训练参数"""
        return self.args_train
        
    @property
    def n_diffusion_steps(self) -> int:
        """获取扩散步数"""
        if self.model is not None and hasattr(self.model, 'n_diffusion_steps'):
            return self.model.n_diffusion_steps
        return self.config.n_diffusion_steps


class DynamicMPDPlanner:
    """
    动态环境MPD规划器
    
    集成预训练MPD模型与动态障碍物处理。
    """
    
    def __init__(
        self,
        model_loader: PretrainedMPDLoader,
        st_sdf=None,
        dynamic_guidance=None,
        tensor_args: Optional[Dict] = None,
    ):
        """
        Args:
            model_loader: 预训练模型加载器
            st_sdf: 时空SDF (可选)
            dynamic_guidance: 动态引导 (可选)
            tensor_args: PyTorch参数
        """
        self.model_loader = model_loader
        self.st_sdf = st_sdf
        self.dynamic_guidance = dynamic_guidance
        
        self.device = model_loader.device
        self.tensor_args = tensor_args or {"device": self.device, "dtype": torch.float32}
        
    def generate_samples(
        self,
        context: Dict[str, torch.Tensor],
        n_samples: int = 10,
        return_all_steps: bool = False,
    ) -> torch.Tensor:
        """
        使用预训练模型生成轨迹样本
        
        Args:
            context: 上下文字典 (起点、终点等)
            n_samples: 采样数量
            return_all_steps: 是否返回所有扩散步骤
            
        Returns:
            生成的轨迹 [n_samples, T, d] 或 [steps, n_samples, T, d]
        """
        model = self.model_loader.get_model()
        adapter = self.model_loader.dataset_adapter
        
        if model is None:
            raise RuntimeError("模型未加载，请先调用 model_loader.load()")
            
        if isinstance(model, GaussianDiffusionModel):
            # 使用数据集适配器构建正确的输入格式
            if adapter is not None:
                print("[INFO] 使用数据集适配器构建模型输入...")
                
                # 获取起点和终点
                start = context.get('start')
                goal = context.get('goal')
                
                if start is not None and goal is not None:
                    # 如果是批次数据，取第一个样本
                    if start.dim() > 1:
                        start = start[0]
                    if goal.dim() > 1:
                        goal = goal[0]
                    
                    # 使用便捷方法准备推理输入
                    hard_conds, context_d = adapter.prepare_inference_inputs(
                        q_start=start,
                        q_goal=goal,
                    )
                    
                    # 获取horizon (从adapter的n_learnable_control_points)
                    horizon = adapter.n_learnable_control_points
                    
                    print(f"[INFO] 调用扩散模型推理: n_samples={n_samples}, horizon={horizon}")
                    print(f"       hard_conds keys: {list(hard_conds.keys())}")
                    print(f"       context_d keys: {list(context_d.keys())}")
                    
                    # 直接调用conditional_sample传入horizon
                    # 注意: context需要预先扩展batch维度
                    context_batched = {}
                    for k, v in context_d.items():
                        context_batched[k] = einops.repeat(v, "... -> b ...", b=n_samples)
                    
                    samples, chain, chain_x_recon = model.conditional_sample(
                        hard_conds=hard_conds,
                        horizon=horizon,
                        batch_size=n_samples,
                        method="ddpm",
                        context_d=context_batched,
                        return_chain=True,
                        return_chain_x_recon=True,
                    )
                    
                    print(f"[INFO] 扩散模型生成成功: samples shape = {samples.shape}")
                    
                    # 反归一化输出
                    samples = adapter.unnormalize_trajectory(samples)
                    
                    if return_all_steps:
                        # 返回整个扩散链
                        chain = einops.rearrange(chain, "b steps ... -> steps b ...")
                        return torch.stack([
                            adapter.unnormalize_trajectory(s) for s in chain
                        ])
                    
                    return samples
                    
            # 回退到旧方法 (无适配器)
            print("[WARN] 无数据集适配器，使用简单context格式")
            samples = model.run_inference(
                context_d=context,
                n_samples=n_samples,
                return_chain=return_all_steps,
                return_chain_x_recon=False,
            )
        elif isinstance(model, CVAEModel):
            # CVAE模型推理
            samples = model.sample(context, n_samples)
        else:
            raise NotImplementedError(f"不支持的模型类型: {type(model)}")
            
        return samples
        
    def apply_dynamic_guidance(
        self,
        trajectories: torch.Tensor,
        n_iterations: int = 20,
        step_size: float = 0.01,
        preserve_endpoints: bool = True,
    ) -> torch.Tensor:
        """
        应用动态障碍物引导优化轨迹
        
        Args:
            trajectories: 初始轨迹 [N, T, d]
            n_iterations: 优化迭代次数
            step_size: 梯度步长
            preserve_endpoints: 是否保持起点和终点不变
            
        Returns:
            优化后的轨迹 [N, T, d]
        """
        if self.dynamic_guidance is None:
            print("[WARN] 未设置动态引导，返回原始轨迹")
            return trajectories
            
        N, T, d = trajectories.shape
        optimized = trajectories.clone()
        
        for i in range(n_iterations):
            total_cost = 0.0
            
            for j in range(N):
                traj = optimized[j]  # [T, d]
                
                # 计算代价和梯度
                costs, gradient = self.dynamic_guidance.compute_total_cost_and_gradient(traj)
                cost = sum(costs.values())
                total_cost += cost.item()
                
                # 应用梯度
                optimized[j] = traj - step_size * gradient
                
                # 保持端点不变
                if preserve_endpoints:
                    optimized[j, 0] = trajectories[j, 0]
                    optimized[j, -1] = trajectories[j, -1]
                    
            if i % 5 == 0:
                print(f"  引导迭代 {i}: 平均代价 = {total_cost/N:.4f}")
                
        return optimized
        
    def plan(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        context: Optional[Dict] = None,
        n_samples: int = 10,
        guidance_iterations: int = 20,
        guidance_step_size: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行完整的动态规划
        
        Args:
            start: 起点 [d]
            goal: 终点 [d]  
            context: 额外上下文
            n_samples: 采样数量
            guidance_iterations: 引导迭代次数
            guidance_step_size: 引导步长
            
        Returns:
            best_trajectory: 最佳轨迹 [T, d]
            info: 规划信息
        """
        info = {
            'n_samples': n_samples,
            'model_loaded': self.model_loader.is_loaded,
        }
        
        # 1. 生成初始轨迹样本
        if self.model_loader.is_loaded:
            print("[PLAN] 使用预训练MPD模型生成轨迹...")
            # 构建上下文
            full_context = {
                'start': start.unsqueeze(0).expand(n_samples, -1),
                'goal': goal.unsqueeze(0).expand(n_samples, -1),
            }
            if context:
                full_context.update(context)
                
            try:
                trajectories = self.generate_samples(full_context, n_samples)
                info['generation_method'] = 'mpd_model'
            except Exception as e:
                print(f"[WARN] MPD模型生成失败: {e}")
                print("[PLAN] 回退到简单插值...")
                trajectories = self._generate_interpolated_trajectories(start, goal, n_samples)
                info['generation_method'] = 'interpolation_fallback'
        else:
            print("[PLAN] MPD模型未加载，使用简单插值...")
            trajectories = self._generate_interpolated_trajectories(start, goal, n_samples)
            info['generation_method'] = 'interpolation'
            
        info['initial_trajectories'] = trajectories.clone()
        
        # 2. 应用动态引导
        if self.dynamic_guidance is not None:
            print(f"[PLAN] 应用动态引导 ({guidance_iterations} 迭代)...")
            trajectories = self.apply_dynamic_guidance(
                trajectories,
                n_iterations=guidance_iterations,
                step_size=guidance_step_size,
            )
            
        # 3. 选择最佳轨迹
        best_idx, costs = self._select_best_trajectory(trajectories)
        best_trajectory = trajectories[best_idx]
        
        info['optimized_trajectories'] = trajectories
        info['all_costs'] = costs
        info['best_idx'] = best_idx
        info['best_cost'] = costs[best_idx]
        
        print(f"[PLAN] 选择轨迹 {best_idx}, 代价 = {costs[best_idx]:.4f}")
        
        return best_trajectory, info
        
    def _generate_interpolated_trajectories(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        n_samples: int,
        n_waypoints: int = 64,
    ) -> torch.Tensor:
        """生成简单插值轨迹 (带随机扰动)"""
        d = start.shape[0]
        
        # 线性插值
        t = torch.linspace(0, 1, n_waypoints, device=self.device)
        base_traj = start.unsqueeze(0) + t.unsqueeze(1) * (goal - start).unsqueeze(0)
        
        trajectories = []
        for i in range(n_samples):
            if i == 0:
                # 第一条是纯直线
                traj = base_traj.clone()
            else:
                # 添加随机扰动
                noise = torch.randn(n_waypoints, d, device=self.device) * 0.1
                noise[0] = 0
                noise[-1] = 0
                traj = base_traj + noise
            trajectories.append(traj)
            
        return torch.stack(trajectories)  # [N, T, d]
        
    def _select_best_trajectory(
        self,
        trajectories: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """选择最佳轨迹"""
        N = trajectories.shape[0]
        costs = torch.zeros(N, device=self.device)
        
        for i in range(N):
            if self.dynamic_guidance is not None:
                cost_dict, _ = self.dynamic_guidance.compute_total_cost_and_gradient(
                    trajectories[i]
                )
                costs[i] = sum(cost_dict.values())
            else:
                # 如果没有引导，用轨迹长度作为代价
                diff = trajectories[i, 1:] - trajectories[i, :-1]
                costs[i] = torch.sum(torch.norm(diff, dim=-1))
                
        best_idx = costs.argmin().item()
        return best_idx, costs


def create_planner_from_config(
    model_dir: str,
    st_sdf=None,
    dynamic_guidance=None,
    device: str = "cuda",
) -> DynamicMPDPlanner:
    """
    从配置创建规划器的便捷函数
    
    Args:
        model_dir: 模型目录路径
        st_sdf: 时空SDF
        dynamic_guidance: 动态引导
        device: 计算设备
        
    Returns:
        配置好的规划器
    """
    config = MPDModelConfig(
        model_dir=model_dir,
        device=device,
    )
    
    loader = PretrainedMPDLoader(config)
    loader.load()
    
    planner = DynamicMPDPlanner(
        model_loader=loader,
        st_sdf=st_sdf,
        dynamic_guidance=dynamic_guidance,
    )
    
    return planner


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("MPD集成模块测试")
    print("=" * 60)
    
    # 测试模型加载
    # 注意: 需要有实际的预训练模型
    model_dir = os.path.expandvars(
        "${HOME}/mpd-build/data_public/EnvSimple2D-RobotPointMass2D/ddpm/bspline"
    )
    
    config = MPDModelConfig(
        model_dir=model_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"\n模型目录: {model_dir}")
    print(f"设备: {config.device}")
    
    loader = PretrainedMPDLoader(config)
    success = loader.load()
    
    if success:
        print("\n模型信息:")
        print(f"  扩散步数: {loader.n_diffusion_steps}")
        print(f"  训练参数: {list(loader.args_train.keys())[:10]}...")
    else:
        print("\n模型加载失败，使用回退模式测试...")
        
    # 创建规划器
    planner = DynamicMPDPlanner(model_loader=loader)
    
    # 测试规划
    start = torch.tensor([-0.8, -0.8], device=config.device)
    goal = torch.tensor([0.8, 0.8], device=config.device)
    
    print(f"\n测试规划: {start.tolist()} -> {goal.tolist()}")
    
    trajectory, info = planner.plan(start, goal, n_samples=5)
    
    print(f"\n规划结果:")
    print(f"  生成方法: {info['generation_method']}")
    print(f"  轨迹形状: {trajectory.shape}")
    print(f"  最佳索引: {info['best_idx']}")
    print(f"  最佳代价: {info['best_cost']:.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

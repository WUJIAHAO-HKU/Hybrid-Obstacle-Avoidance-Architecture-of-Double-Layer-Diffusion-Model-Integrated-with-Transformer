"""
层级扩散规划器

将上层轨迹预测扩散模型与下层MPD运动规划模型串联，实现动态环境下的避障规划。

架构:
    1. 上层: 轨迹预测扩散模型 - 预测障碍物未来轨迹
    2. 中层: 时空SDF生成器 - 将预测轨迹转换为时空距离场
    3. 下层: MPD运动规划 - 基于时空SDF生成避障轨迹

Author: Dynamic MPD Project
Date: 2026-01-25
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

# 添加路径
DYNAMIC_MPD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DYNAMIC_MPD_ROOT)

from src.spatiotemporal_sdf import SpatioTemporalSDF, ObstacleTrajectory
from src.dynamic_guidance import DynamicCollisionGuidance


@dataclass
class HierarchicalPlannerConfig:
    """层级规划器配置"""
    # 轨迹预测模型配置
    trajectory_model_path: str = ""
    obs_len: int = 8
    pred_len: int = 12
    d_model: int = 128
    encoder_layers: int = 4
    denoiser_hidden: int = 256
    denoiser_layers: int = 4
    num_diffusion_steps: int = 100
    
    # 时空SDF配置 - 高分辨率
    workspace_bounds: List[float] = None  # [x_min, x_max, y_min, y_max]
    spatial_resolution: float = 0.02  # 提高空间分辨率 (0.05 -> 0.02)
    time_horizon: float = 4.8  # pred_len * dt
    num_time_steps: int = 48  # 提高时间分辨率 (12 -> 48)
    
    # MPD规划配置
    mpd_model_path: str = ""
    n_trajectory_samples: int = 16  # 增加采样数 (10 -> 16)
    guidance_iterations: int = 30  # 增加引导迭代 (20 -> 30)
    guidance_step_size: float = 0.02  # 增加引导步长 (0.01 -> 0.02)
    
    # 障碍物配置
    obstacle_radius: float = 0.3  # 行人半径
    safety_margin: float = 0.1   # 安全边距
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.workspace_bounds is None:
            self.workspace_bounds = [-10.0, 10.0, -10.0, 10.0]


class TrajectoryPredictionModule:
    """
    轨迹预测模块
    
    封装训练好的轨迹预测扩散模型，提供预测接口。
    """
    
    def __init__(
        self,
        model_path: str,
        config: HierarchicalPlannerConfig,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.model = None
        self.is_loaded = False
        
    def load(self) -> bool:
        """加载预训练模型"""
        if not os.path.exists(self.model_path):
            print(f"[ERROR] 轨迹预测模型不存在: {self.model_path}")
            return False
            
        try:
            # 延迟导入避免循环依赖
            from src.trajectory_diffusion import ObstacleTrajectoryDiffusion
            
            # 创建模型
            self.model = ObstacleTrajectoryDiffusion(
                obs_len=self.config.obs_len,
                pred_len=self.config.pred_len,
                traj_dim=2,
                d_model=self.config.d_model,
                encoder_layers=self.config.encoder_layers,
                denoiser_hidden=self.config.denoiser_hidden,
                denoiser_layers=self.config.denoiser_layers,
                num_diffusion_steps=self.config.num_diffusion_steps,
            ).to(self.device)
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            self.is_loaded = True
            print(f"[INFO] 轨迹预测模型加载成功: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 加载轨迹预测模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _smooth_trajectory(self, trajectory: torch.Tensor, window_size: int = 3) -> torch.Tensor:
        """
        对轨迹进行平滑处理
        
        Args:
            trajectory: [pred_len, 2] 轨迹
            window_size: 平滑窗口大小
            
        Returns:
            smoothed: [pred_len, 2] 平滑后的轨迹
        """
        if trajectory.shape[0] <= window_size:
            return trajectory
            
        # 使用简单移动平均进行平滑
        padded = torch.cat([
            trajectory[:1].repeat(window_size // 2, 1),
            trajectory,
            trajectory[-1:].repeat(window_size // 2, 1),
        ], dim=0)
        
        kernel = torch.ones(window_size, device=trajectory.device) / window_size
        smoothed = torch.zeros_like(trajectory)
        for i in range(trajectory.shape[0]):
            smoothed[i] = padded[i:i+window_size].mean(dim=0)
            
        return smoothed
    
    def _select_best_prediction(
        self,
        predictions: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """
        从多样本预测中选择最佳轨迹
        
        选择标准:
        1. 速度一致性 (与历史轨迹末端速度相近)
        2. 平滑度 (加速度变化小)
        3. 物理可行性 (速度不超过阈值)
        
        Args:
            predictions: [num_samples, pred_len, 2] 预测轨迹
            history: [obs_len, 2] 历史轨迹
            
        Returns:
            best_prediction: [pred_len, 2]
        """
        num_samples = predictions.shape[0]
        
        if num_samples == 1:
            return predictions[0]
        
        # 计算历史速度
        hist_vel = history[-1] - history[-2]  # [2]
        
        scores = []
        for i in range(num_samples):
            pred = predictions[i]  # [pred_len, 2]
            
            # 1. 速度一致性: 预测起始速度与历史末端速度的差异
            pred_start_vel = pred[0] - history[-1]
            vel_consistency = -torch.norm(pred_start_vel - hist_vel).item()
            
            # 2. 平滑度: 加速度变化小
            velocities = pred[1:] - pred[:-1]
            accelerations = velocities[1:] - velocities[:-1]
            smoothness = -torch.norm(accelerations, dim=-1).mean().item()
            
            # 3. 物理可行性: 速度不超过阈值
            speeds = torch.norm(velocities, dim=-1)
            max_speed = speeds.max().item()
            feasibility = -max(0, max_speed - 0.5)  # 假设最大速度 0.5 m/step
            
            # 综合评分
            score = vel_consistency * 2.0 + smoothness * 1.0 + feasibility * 3.0
            scores.append(score)
            
        best_idx = np.argmax(scores)
        return predictions[best_idx]
    
    @torch.no_grad()
    def predict(
        self,
        history_trajectories: torch.Tensor,
        neighbor_trajectories: Optional[torch.Tensor] = None,
        num_samples: int = 5,  # 增加默认样本数
        use_best_selection: bool = True,  # 使用最佳选择
        use_smoothing: bool = True,  # 使用平滑
    ) -> torch.Tensor:
        """
        预测障碍物未来轨迹 (增强版)
        
        Args:
            history_trajectories: [num_obstacles, obs_len, 2] 历史轨迹
            neighbor_trajectories: [num_obstacles, num_neighbors, obs_len, 2] 邻居轨迹
            num_samples: 每个障碍物生成的样本数
            use_best_selection: 是否使用最佳轨迹选择
            use_smoothing: 是否对轨迹进行平滑
            
        Returns:
            predicted_trajectories: [num_obstacles, num_samples, pred_len, 2] 或
                                    [num_obstacles, pred_len, 2] (如果use_best_selection=True)
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
            
        num_obstacles = history_trajectories.shape[0]
        all_predictions = []
        
        for i in range(num_obstacles):
            obs_history = history_trajectories[i:i+1]  # [1, obs_len, 2]
            
            # 获取邻居轨迹 (其他障碍物作为邻居)
            other_indices = [j for j in range(num_obstacles) if j != i]
            if len(other_indices) > 0:
                obs_neighbors = history_trajectories[other_indices].unsqueeze(0)  # [1, num_neighbors, obs_len, 2]
            else:
                obs_neighbors = None
                
            # 使用模型采样
            # 模型返回 [batch, n_samples, pred_len, traj_dim]
            pred = self.model.sample(
                obs_history,
                neighbors_history=obs_neighbors,
                n_samples=num_samples,
            )
            # pred shape: [1, num_samples, pred_len, 2]
            pred = pred[0]  # [num_samples, pred_len, 2]
            
            if use_best_selection:
                # 选择最佳预测
                best_pred = self._select_best_prediction(pred, history_trajectories[i])
                if use_smoothing:
                    best_pred = self._smooth_trajectory(best_pred)
                all_predictions.append(best_pred)  # [pred_len, 2]
            else:
                all_predictions.append(pred)  # [num_samples, pred_len, 2]
            
        if use_best_selection:
            return torch.stack(all_predictions)  # [num_obstacles, pred_len, 2]
        else:
            return torch.stack(all_predictions)  # [num_obstacles, num_samples, pred_len, 2]
    
    @torch.no_grad()
    def predict_batch(
        self,
        history_trajectories: torch.Tensor,
        neighbor_trajectories: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        批量预测 (单样本)
        
        Args:
            history_trajectories: [batch, obs_len, 2]
            neighbor_trajectories: [batch, num_neighbors, obs_len, 2]
            
        Returns:
            predictions: [batch, pred_len, 2]
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        # 返回shape [batch, 1, pred_len, 2]，取第一个样本
        preds = self.model.sample(
            history_trajectories,
            neighbors_history=neighbor_trajectories,
            n_samples=1,
        )
        return preds[:, 0]  # [batch, pred_len, 2]


class HierarchicalDiffusionPlanner:
    """
    层级扩散规划器
    
    整合轨迹预测、时空SDF和运动规划的完整管线。
    """
    
    def __init__(
        self,
        config: HierarchicalPlannerConfig,
    ):
        self.config = config
        self.device = config.device
        
        # 轨迹预测模块
        self.trajectory_predictor: Optional[TrajectoryPredictionModule] = None
        
        # 时空SDF
        self.st_sdf: Optional[SpatioTemporalSDF] = None
        
        # 动态引导
        self.dynamic_guidance: Optional[DynamicCollisionGuidance] = None
        
        # MPD规划器 (可选)
        self.mpd_planner = None
        
        # 状态
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """初始化所有组件"""
        success = True
        
        # 1. 加载轨迹预测模型
        if self.config.trajectory_model_path:
            self.trajectory_predictor = TrajectoryPredictionModule(
                self.config.trajectory_model_path,
                self.config,
                self.device,
            )
            if not self.trajectory_predictor.load():
                print("[WARN] 轨迹预测模型加载失败，将使用线性预测")
                self.trajectory_predictor = None
        else:
            print("[INFO] 未指定轨迹预测模型，将使用线性预测")
            
        # 2. 创建时空SDF
        self.st_sdf = SpatioTemporalSDF(
            workspace_bounds=self.config.workspace_bounds,
            spatial_resolution=self.config.spatial_resolution,
            time_horizon=self.config.time_horizon,
            num_time_steps=self.config.num_time_steps,
            device=self.device,
        )
        print(f"[INFO] 时空SDF创建成功: {self.st_sdf}")
        
        # 3. 创建动态引导
        self.dynamic_guidance = DynamicCollisionGuidance(
            st_sdf=self.st_sdf,
            safety_margin=self.config.safety_margin,
            device=self.device,
        )
        print("[INFO] 动态引导创建成功")
        
        # 4. 加载MPD模型 (可选)
        if self.config.mpd_model_path and os.path.exists(self.config.mpd_model_path):
            try:
                from src.mpd_integration import PretrainedMPDLoader, MPDModelConfig, DynamicMPDPlanner
                
                mpd_config = MPDModelConfig(
                    model_dir=self.config.mpd_model_path,
                    device=self.device,
                )
                loader = PretrainedMPDLoader(mpd_config)
                if loader.load():
                    self.mpd_planner = DynamicMPDPlanner(
                        model_loader=loader,
                        st_sdf=self.st_sdf,
                        dynamic_guidance=self.dynamic_guidance,
                    )
                    print("[INFO] MPD规划器加载成功")
                else:
                    print("[WARN] MPD模型加载失败")
            except Exception as e:
                print(f"[WARN] MPD集成失败: {e}")
        else:
            print("[INFO] 未指定MPD模型，将使用简单轨迹生成")
            
        self.is_initialized = True
        return success
    
    def predict_obstacle_trajectories(
        self,
        obstacle_histories: torch.Tensor,
        neighbor_histories: Optional[torch.Tensor] = None,
        num_samples: int = 5,  # 增加默认样本数
        force_linear: bool = None,  # 强制使用线性预测
    ) -> List[ObstacleTrajectory]:
        """
        预测障碍物轨迹并转换为ObstacleTrajectory格式 (自适应版)
        
        自动检测运动模式并选择最佳预测方法:
        - 线性运动 → 恒速预测 (更准确)
        - 非线性运动 → 扩散模型预测 (如果可用)
        
        Args:
            obstacle_histories: [num_obstacles, obs_len, 2] 障碍物历史轨迹
            neighbor_histories: [num_obstacles, num_neighbors, obs_len, 2]
            num_samples: 预测样本数 (用于多样本选择最佳)
            force_linear: 强制使用线性预测 (None=自动检测)
            
        Returns:
            obstacle_trajectories: ObstacleTrajectory列表
        """
        num_obstacles = obstacle_histories.shape[0]
        
        # 自动检测是否应该使用线性预测
        if force_linear is None:
            use_linear = self._should_use_linear_prediction(obstacle_histories)
        else:
            use_linear = force_linear
        
        if use_linear:
            # 对于线性运动，使用恒速预测 (更准确!)
            print("[PREDICT] 检测到线性运动，使用恒速预测...")
            predicted_futures = self._linear_predict(obstacle_histories, use_smoothed_velocity=True)
        elif self.trajectory_predictor is not None and self.trajectory_predictor.is_loaded:
            # 对于非线性运动，尝试使用扩散模型
            print(f"[PREDICT] 检测到非线性运动，使用扩散模型 (采样数={num_samples})...")
            try:
                predicted_futures = self.trajectory_predictor.predict(
                    obstacle_histories,
                    neighbor_histories,
                    num_samples=num_samples,
                    use_best_selection=True,
                    use_smoothing=True,
                )
            except Exception as e:
                print(f"[WARN] 扩散模型预测失败: {e}，回退到恒速预测")
                predicted_futures = self._linear_predict(obstacle_histories)
        else:
            # 回退到线性预测
            print("[PREDICT] 使用恒速预测（默认）...")
            predicted_futures = self._linear_predict(obstacle_histories)
            
        # 转换为ObstacleTrajectory格式
        obstacle_trajs = []
        dt = self.config.time_horizon / self.config.num_time_steps
        
        for i in range(num_obstacles):
            positions = predicted_futures[i]  # [pred_len, 2]
            timestamps = torch.arange(
                self.config.pred_len, 
                device=self.device
            ).float() * dt
            
            obs_traj = ObstacleTrajectory(
                positions=positions,
                radii=self.config.obstacle_radius,
                timestamps=timestamps,
                obstacle_id=i,
            )
            obstacle_trajs.append(obs_traj)
            
        return obstacle_trajs
    
    def _linear_predict(
        self,
        histories: torch.Tensor,
        use_smoothed_velocity: bool = True,
    ) -> torch.Tensor:
        """
        恒速线性预测 (改进版)
        
        这是最适合当前测试场景的预测方法，因为:
        1. 测试场景中的障碍物是线性运动
        2. 扩散模型在不同尺度的数据上训练，无法泛化
        
        Args:
            histories: [num_obstacles, obs_len, 2]
            use_smoothed_velocity: 是否使用平滑速度估计
            
        Returns:
            predictions: [num_obstacles, pred_len, 2]
        """
        if use_smoothed_velocity and histories.shape[1] >= 3:
            # 使用多帧速度平均以减少噪声
            recent_vels = histories[:, -3:] - histories[:, -4:-1]  # [N, 3, 2]
            velocities = recent_vels.mean(dim=1)  # [N, 2]
        else:
            # 使用最后两帧估计速度
            velocities = histories[:, -1] - histories[:, -2]  # [num_obstacles, 2]
        
        # 线性外推
        predictions = []
        for t in range(self.config.pred_len):
            pos = histories[:, -1] + (t + 1) * velocities
            predictions.append(pos)
            
        return torch.stack(predictions, dim=1)  # [num_obstacles, pred_len, 2]
    
    def _should_use_linear_prediction(
        self,
        histories: torch.Tensor,
        linearity_threshold: float = 0.15,
    ) -> bool:
        """
        判断是否应该使用线性预测
        
        通过分析历史轨迹的速度变化来判断运动是否为线性
        
        Args:
            histories: [num_obstacles, obs_len, 2]
            linearity_threshold: 线性度阈值
            
        Returns:
            should_use_linear: 是否使用线性预测
        """
        # 计算速度序列
        velocities = histories[:, 1:] - histories[:, :-1]  # [N, T-1, 2]
        
        # 计算速度的标准差和均值
        vel_std = velocities.std(dim=1)  # [N, 2]
        vel_mean = velocities.mean(dim=1).abs() + 1e-6  # [N, 2]
        
        # 计算变异系数 (CV)
        cv = vel_std / vel_mean  # [N, 2]
        
        # 如果所有障碍物的速度变化都很小，则使用线性预测
        max_cv = cv.max().item()
        
        return max_cv < linearity_threshold
    
    def update_spatiotemporal_sdf(
        self,
        obstacle_trajectories: List[ObstacleTrajectory],
    ):
        """
        更新时空SDF
        
        Args:
            obstacle_trajectories: 障碍物轨迹列表
        """
        print(f"[SDF] 更新时空SDF ({len(obstacle_trajectories)} 个障碍物)...")
        self.st_sdf.compute_from_obstacle_trajectories(obstacle_trajectories)
        
    def generate_robot_trajectory(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        生成机器人避障轨迹
        
        Args:
            start: [2] 起点
            goal: [2] 终点
            n_samples: 轨迹样本数
            
        Returns:
            best_trajectory: [T, 2] 最佳轨迹
            info: 规划信息
        """
        info = {}
        
        if self.mpd_planner is not None:
            # 使用MPD规划
            print("[PLAN] 使用MPD模型规划...")
            trajectory, mpd_info = self.mpd_planner.plan(
                start, goal,
                n_samples=n_samples,
                guidance_iterations=self.config.guidance_iterations,
                guidance_step_size=self.config.guidance_step_size,
            )
            info.update(mpd_info)
            info['method'] = 'mpd'
        else:
            # 使用简单规划 + 引导优化
            print("[PLAN] 使用简单规划 + 引导优化...")
            trajectories = self._generate_initial_trajectories(
                start, goal, n_samples
            )
            
            # 应用动态引导优化
            trajectories = self._apply_guidance(trajectories)
            
            # 选择最佳轨迹
            best_idx, costs = self._select_best_trajectory(trajectories)
            trajectory = trajectories[best_idx]
            
            info['initial_trajectories'] = self._generate_initial_trajectories(start, goal, n_samples)
            info['optimized_trajectories'] = trajectories
            info['best_idx'] = best_idx
            info['costs'] = costs
            info['method'] = 'guided_interpolation'
            
        return trajectory, info
    
    def _generate_initial_trajectories(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        n_samples: int,
        n_waypoints: int = 64,
    ) -> torch.Tensor:
        """生成初始轨迹样本"""
        d = start.shape[0]
        
        # 线性插值基础
        t = torch.linspace(0, 1, n_waypoints, device=self.device)
        base_traj = start.unsqueeze(0) + t.unsqueeze(1) * (goal - start).unsqueeze(0)
        
        trajectories = []
        for i in range(n_samples):
            if i == 0:
                traj = base_traj.clone()
            else:
                # 添加Bezier风格随机扰动
                noise_scale = 0.2 * torch.sin(t * np.pi).unsqueeze(1)  # 中间扰动大
                noise = torch.randn(n_waypoints, d, device=self.device) * noise_scale
                noise[0] = 0
                noise[-1] = 0
                traj = base_traj + noise
            trajectories.append(traj)
            
        return torch.stack(trajectories)
    
    def _apply_guidance(
        self,
        trajectories: torch.Tensor,
        n_iterations: Optional[int] = None,
        step_size: Optional[float] = None,
    ) -> torch.Tensor:
        """应用动态引导优化"""
        n_iterations = n_iterations or self.config.guidance_iterations
        step_size = step_size or self.config.guidance_step_size
        
        N, T, d = trajectories.shape
        optimized = trajectories.clone()
        
        for i in range(n_iterations):
            total_cost = 0.0
            
            for j in range(N):
                traj = optimized[j]
                costs, gradient = self.dynamic_guidance.compute_total_cost_and_gradient(traj)
                cost = sum(costs.values())
                total_cost += cost.item()
                
                # 梯度下降
                optimized[j] = traj - step_size * gradient
                
                # 保持端点
                optimized[j, 0] = trajectories[j, 0]
                optimized[j, -1] = trajectories[j, -1]
                
            if i % 5 == 0:
                print(f"  引导迭代 {i}: 平均代价 = {total_cost/N:.4f}")
                
        return optimized
    
    def _select_best_trajectory(
        self,
        trajectories: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """选择最佳轨迹"""
        N = trajectories.shape[0]
        costs = torch.zeros(N, device=self.device)
        
        for i in range(N):
            cost_dict, _ = self.dynamic_guidance.compute_total_cost_and_gradient(trajectories[i])
            costs[i] = sum(cost_dict.values())
            
        best_idx = costs.argmin().item()
        return best_idx, costs
    
    def plan(
        self,
        robot_start: torch.Tensor,
        robot_goal: torch.Tensor,
        obstacle_histories: torch.Tensor,
        neighbor_histories: Optional[torch.Tensor] = None,
        num_prediction_samples: int = 1,
        num_trajectory_samples: int = 10,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        完整的层级规划流程
        
        Args:
            robot_start: [2] 机器人起点
            robot_goal: [2] 机器人终点
            obstacle_histories: [num_obstacles, obs_len, 2] 障碍物历史轨迹
            neighbor_histories: [num_obstacles, num_neighbors, obs_len, 2]
            num_prediction_samples: 轨迹预测采样数
            num_trajectory_samples: 规划轨迹采样数
            
        Returns:
            robot_trajectory: [T, 2] 机器人轨迹
            info: 完整信息
        """
        if not self.is_initialized:
            raise RuntimeError("规划器未初始化")
            
        info = {}
        
        # Step 1: 预测障碍物轨迹
        print("\n" + "="*50)
        print("Step 1: 预测障碍物轨迹")
        print("="*50)
        obstacle_trajs = self.predict_obstacle_trajectories(
            obstacle_histories,
            neighbor_histories,
            num_prediction_samples,
        )
        info['obstacle_trajectories'] = obstacle_trajs
        info['num_obstacles'] = len(obstacle_trajs)
        
        # Step 2: 更新时空SDF
        print("\n" + "="*50)
        print("Step 2: 更新时空SDF")
        print("="*50)
        self.update_spatiotemporal_sdf(obstacle_trajs)
        info['st_sdf'] = self.st_sdf
        
        # Step 3: 生成避障轨迹
        print("\n" + "="*50)
        print("Step 3: 生成机器人避障轨迹")
        print("="*50)
        robot_trajectory, plan_info = self.generate_robot_trajectory(
            robot_start,
            robot_goal,
            num_trajectory_samples,
        )
        info.update(plan_info)
        
        # 评估轨迹安全性
        sdf_values = self.st_sdf.query_trajectory(robot_trajectory)
        min_sdf = sdf_values.min().item()
        info['min_sdf'] = min_sdf
        info['collision_free'] = min_sdf > 0
        
        print(f"\n规划结果:")
        print(f"  轨迹长度: {robot_trajectory.shape[0]}")
        print(f"  最小SDF: {min_sdf:.4f}")
        print(f"  无碰撞: {info['collision_free']}")
        
        return robot_trajectory, info


def create_hierarchical_planner(
    trajectory_model_path: str = "",
    mpd_model_path: str = "",
    workspace_bounds: List[float] = None,
    device: str = "cuda",
) -> HierarchicalDiffusionPlanner:
    """
    创建层级规划器的便捷函数
    
    Args:
        trajectory_model_path: 轨迹预测模型路径
        mpd_model_path: MPD模型路径
        workspace_bounds: 工作空间边界
        device: 设备
        
    Returns:
        planner: 初始化好的规划器
    """
    config = HierarchicalPlannerConfig(
        trajectory_model_path=trajectory_model_path,
        mpd_model_path=mpd_model_path,
        workspace_bounds=workspace_bounds,
        device=device,
    )
    
    planner = HierarchicalDiffusionPlanner(config)
    planner.initialize()
    
    return planner


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("层级扩散规划器测试")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 创建配置
    config = HierarchicalPlannerConfig(
        workspace_bounds=[-5.0, 5.0, -5.0, 5.0],
        device=device,
    )
    
    # 创建规划器
    planner = HierarchicalDiffusionPlanner(config)
    planner.initialize()
    
    # 测试数据
    robot_start = torch.tensor([-3.0, -3.0], device=device)
    robot_goal = torch.tensor([3.0, 3.0], device=device)
    
    # 模拟障碍物历史轨迹
    obs_len = 8
    num_obstacles = 3
    obstacle_histories = torch.zeros(num_obstacles, obs_len, 2, device=device)
    
    # 障碍物1: 向右移动
    for t in range(obs_len):
        obstacle_histories[0, t] = torch.tensor([-1.0 + t*0.2, 0.0])
        
    # 障碍物2: 向下移动
    for t in range(obs_len):
        obstacle_histories[1, t] = torch.tensor([1.0, 2.0 - t*0.2])
        
    # 障碍物3: 静止
    for t in range(obs_len):
        obstacle_histories[2, t] = torch.tensor([0.0, -1.5])
    
    # 执行规划
    trajectory, info = planner.plan(
        robot_start,
        robot_goal,
        obstacle_histories,
        num_trajectory_samples=5,
    )
    
    print("\n" + "="*60)
    print("规划完成!")
    print(f"轨迹形状: {trajectory.shape}")
    print(f"最小SDF: {info['min_sdf']:.4f}")
    print(f"无碰撞: {info['collision_free']}")
    print("="*60)

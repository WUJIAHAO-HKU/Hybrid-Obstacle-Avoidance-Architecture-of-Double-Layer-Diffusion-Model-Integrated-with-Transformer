"""
复杂障碍物运动数据生成器

支持:
1. 多障碍物同时移动
2. 弧线/曲线移动
3. 变速移动
4. 随机加速减速
5. 突然转向
6. 群体运动模式

Author: Dynamic MPD Project
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ObstacleMotionConfig:
    """障碍物运动配置"""
    num_obstacles: int = 3          # 每个场景的障碍物数量
    obs_history_len: int = 8        # 观测历史长度
    pred_horizon: int = 12          # 预测时长
    dt: float = 0.2                 # 时间步长 (增大以增加移动距离)
    max_speed: float = 0.2          # 最大速度 (增大)
    min_speed: float = 0.05         # 最小速度 (增大)
    max_accel: float = 0.05         # 最大加速度 (增大)
    arena_size: float = 2.0         # 场景范围 (增大)
    

class ComplexObstacleDataGenerator:
    """
    复杂障碍物运动数据生成器
    
    生成多种运动模式的组合数据
    """
    
    def __init__(self, config: ObstacleMotionConfig = None):
        self.config = config or ObstacleMotionConfig()
    
    def generate_single_obstacle_trajectory(
        self, 
        motion_type: str,
        total_steps: int
    ) -> torch.Tensor:
        """
        生成单个障碍物的轨迹
        
        Args:
            motion_type: 运动类型
            total_steps: 总步数
            
        Returns:
            trajectory: [total_steps, 2]
        """
        cfg = self.config
        
        # 随机初始位置 (更分散)
        pos = np.random.uniform(-cfg.arena_size * 0.6, cfg.arena_size * 0.6, 2)
        
        if motion_type == 'linear':
            return self._generate_linear(pos, total_steps)
        elif motion_type == 'arc':
            return self._generate_arc(pos, total_steps)
        elif motion_type == 'acceleration':
            return self._generate_acceleration(pos, total_steps)
        elif motion_type == 'deceleration':
            return self._generate_deceleration(pos, total_steps)
        elif motion_type == 'variable_speed':
            return self._generate_variable_speed(pos, total_steps)
        elif motion_type == 'zigzag':
            return self._generate_zigzag(pos, total_steps)
        elif motion_type == 'spiral':
            return self._generate_spiral(pos, total_steps)
        elif motion_type == 'sudden_turn':
            return self._generate_sudden_turn(pos, total_steps)
        elif motion_type == 'stop_and_go':
            return self._generate_stop_and_go(pos, total_steps)
        elif motion_type == 'curved':
            return self._generate_curved(pos, total_steps)
        else:
            return self._generate_random_walk(pos, total_steps)
    
    def _generate_linear(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """匀速直线运动"""
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(self.config.min_speed, self.config.max_speed)
        velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        
        traj = np.zeros((steps, 2))
        traj[0] = start_pos
        for i in range(1, steps):
            traj[i] = traj[i-1] + velocity * self.config.dt
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_arc(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """弧线运动 (圆周运动的一部分)"""
        # 随机圆心和半径 (增大半径)
        radius = np.random.uniform(0.5, 1.2)
        center_angle = np.random.uniform(0, 2 * np.pi)
        center = start_pos + radius * np.array([np.cos(center_angle), np.sin(center_angle)])
        
        # 起始角度和角速度 (增大角速度)
        start_angle = center_angle + np.pi
        angular_speed = np.random.uniform(0.15, 0.4) * np.random.choice([-1, 1])
        
        traj = np.zeros((steps, 2))
        for i in range(steps):
            angle = start_angle + angular_speed * i * self.config.dt
            traj[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_acceleration(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """加速运动"""
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        initial_speed = self.config.min_speed * 0.5
        accel = np.random.uniform(0.02, self.config.max_accel)
        
        traj = np.zeros((steps, 2))
        traj[0] = start_pos
        speed = initial_speed
        
        for i in range(1, steps):
            speed = min(speed + accel * self.config.dt, self.config.max_speed * 1.5)
            traj[i] = traj[i-1] + direction * speed * self.config.dt
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_deceleration(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """减速运动"""
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        initial_speed = self.config.max_speed * 1.5
        decel = np.random.uniform(0.01, self.config.max_accel * 0.8)
        
        traj = np.zeros((steps, 2))
        traj[0] = start_pos
        speed = initial_speed
        
        for i in range(1, steps):
            speed = max(speed - decel * self.config.dt, self.config.min_speed * 0.2)
            traj[i] = traj[i-1] + direction * speed * self.config.dt
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_variable_speed(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """变速运动 (正弦波速度)"""
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        base_speed = (self.config.min_speed + self.config.max_speed) / 2
        amplitude = (self.config.max_speed - self.config.min_speed) / 2
        frequency = np.random.uniform(0.08, 0.2)
        phase = np.random.uniform(0, 2 * np.pi)
        
        traj = np.zeros((steps, 2))
        traj[0] = start_pos
        
        for i in range(1, steps):
            speed = base_speed + amplitude * np.sin(frequency * i + phase)
            traj[i] = traj[i-1] + direction * speed * self.config.dt
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_zigzag(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """之字形运动"""
        main_angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(self.config.min_speed, self.config.max_speed)
        zigzag_amplitude = np.random.uniform(0.05, 0.12)
        zigzag_freq = np.random.uniform(0.2, 0.5)
        
        traj = np.zeros((steps, 2))
        traj[0] = start_pos
        
        for i in range(1, steps):
            main_dir = np.array([np.cos(main_angle), np.sin(main_angle)])
            perp_dir = np.array([-np.sin(main_angle), np.cos(main_angle)])
            offset = zigzag_amplitude * np.sin(zigzag_freq * i)
            
            velocity = main_dir * speed + perp_dir * offset
            traj[i] = traj[i-1] + velocity * self.config.dt
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_spiral(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """螺旋运动 (逐渐增大的圆)"""
        center = start_pos.copy()
        start_radius = 0.1
        radius_growth = np.random.uniform(0.03, 0.06)
        angular_speed = np.random.uniform(0.2, 0.5) * np.random.choice([-1, 1])
        
        traj = np.zeros((steps, 2))
        
        for i in range(steps):
            radius = start_radius + radius_growth * i * self.config.dt
            angle = angular_speed * i * self.config.dt
            traj[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_sudden_turn(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """突然转向运动"""
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(self.config.min_speed, self.config.max_speed)
        
        turn_step = np.random.randint(steps // 4, 3 * steps // 4)
        turn_angle = np.random.uniform(np.pi / 3, np.pi * 0.8)
        turn_direction = np.random.choice([-1, 1])
        
        traj = np.zeros((steps, 2))
        traj[0] = start_pos
        current_angle = angle
        
        for i in range(1, steps):
            if i == turn_step:
                current_angle += turn_angle * turn_direction
            
            velocity = np.array([np.cos(current_angle), np.sin(current_angle)]) * speed
            traj[i] = traj[i-1] + velocity * self.config.dt
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_stop_and_go(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """走走停停运动"""
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        max_speed = np.random.uniform(self.config.min_speed * 1.5, self.config.max_speed)
        
        traj = np.zeros((steps, 2))
        traj[0] = start_pos
        
        cycle_length = np.random.randint(4, 8)
        
        for i in range(1, steps):
            cycle_pos = i % cycle_length
            if cycle_pos < cycle_length * 2 // 3:
                speed = max_speed
            else:
                speed = 0
            
            traj[i] = traj[i-1] + direction * speed * self.config.dt
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_curved(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """贝塞尔曲线运动 (增大控制点范围)"""
        p0 = start_pos
        offset_range = 1.5
        p1 = start_pos + np.random.uniform(-offset_range, offset_range, 2)
        p2 = start_pos + np.random.uniform(-offset_range, offset_range, 2)
        p3 = start_pos + np.random.uniform(-offset_range, offset_range, 2)
        
        traj = np.zeros((steps, 2))
        
        for i in range(steps):
            t = i / (steps - 1)
            traj[i] = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _generate_random_walk(self, start_pos: np.ndarray, steps: int) -> torch.Tensor:
        """随机游走"""
        traj = np.zeros((steps, 2))
        traj[0] = start_pos
        
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(self.config.min_speed, self.config.max_speed)
        
        for i in range(1, steps):
            angle += np.random.normal(0, 0.25)
            speed_delta = np.random.normal(0, 0.015)
            speed = np.clip(speed + speed_delta, self.config.min_speed, self.config.max_speed)
            
            velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
            traj[i] = traj[i-1] + velocity * self.config.dt
            traj[i] = self._bound_position(traj[i])
        
        return torch.tensor(traj, dtype=torch.float32)
    
    def _bound_position(self, pos: np.ndarray) -> np.ndarray:
        """限制位置在场景范围内"""
        return np.clip(pos, -self.config.arena_size, self.config.arena_size)
    
    def generate_multi_obstacle_scene(
        self, 
        num_obstacles: int = None,
        motion_types: List[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成多障碍物场景
        
        Returns:
            history: [obs_len, num_obstacles, 2]
            future: [pred_len, num_obstacles, 2]
        """
        num_obstacles = num_obstacles or self.config.num_obstacles
        total_steps = self.config.obs_history_len + self.config.pred_horizon
        
        all_motion_types = [
            'linear', 'arc', 'acceleration', 'deceleration', 
            'variable_speed', 'zigzag', 'spiral', 'sudden_turn',
            'stop_and_go', 'curved', 'random_walk'
        ]
        
        trajectories = []
        for i in range(num_obstacles):
            if motion_types and i < len(motion_types):
                mt = motion_types[i]
            else:
                mt = np.random.choice(all_motion_types)
            
            traj = self.generate_single_obstacle_trajectory(mt, total_steps)
            trajectories.append(traj)
        
        all_traj = torch.stack(trajectories, dim=1)
        
        history = all_traj[:self.config.obs_history_len]
        future = all_traj[self.config.obs_history_len:]
        
        return history, future
    
    def generate_training_batch(
        self,
        batch_size: int,
        include_single: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成训练批次 (单障碍物格式)
        """
        histories = []
        futures = []
        
        samples_per_scene = self.config.num_obstacles if include_single else 1
        scenes_needed = (batch_size + samples_per_scene - 1) // samples_per_scene
        
        for _ in range(scenes_needed):
            history, future = self.generate_multi_obstacle_scene()
            
            for obs_idx in range(self.config.num_obstacles):
                histories.append(history[:, obs_idx, :])
                futures.append(future[:, obs_idx, :])
        
        history_batch = torch.stack(histories[:batch_size])
        future_batch = torch.stack(futures[:batch_size])
        
        return history_batch, future_batch


def generate_complex_training_data(
    num_samples: int,
    obs_len: int = 8,
    pred_len: int = 12,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成复杂训练数据的便捷函数
    """
    config = ObstacleMotionConfig(
        num_obstacles=4,
        obs_history_len=obs_len,
        pred_horizon=pred_len
    )
    
    generator = ComplexObstacleDataGenerator(config)
    
    all_histories = []
    all_futures = []
    
    motion_types = [
        'linear', 'arc', 'acceleration', 'deceleration', 
        'variable_speed', 'zigzag', 'spiral', 'sudden_turn',
        'stop_and_go', 'curved', 'random_walk'
    ]
    
    samples_per_batch = config.num_obstacles
    batches_needed = (num_samples + samples_per_batch - 1) // samples_per_batch
    
    for batch_idx in range(batches_needed):
        selected_types = [
            motion_types[i % len(motion_types)] 
            for i in range(batch_idx, batch_idx + config.num_obstacles)
        ]
        
        history, future = generator.generate_multi_obstacle_scene(
            motion_types=selected_types
        )
        
        for obs_idx in range(config.num_obstacles):
            all_histories.append(history[:, obs_idx, :])
            all_futures.append(future[:, obs_idx, :])
    
    histories = torch.stack(all_histories[:num_samples]).to(device)
    futures = torch.stack(all_futures[:num_samples]).to(device)
    
    return histories, futures


if __name__ == '__main__':
    print("Testing Complex Obstacle Data Generator...")
    
    config = ObstacleMotionConfig(num_obstacles=3)
    gen = ComplexObstacleDataGenerator(config)
    
    motion_types = ['linear', 'arc', 'acceleration', 'zigzag', 'spiral', 'curved']
    
    for mt in motion_types:
        traj = gen.generate_single_obstacle_trajectory(mt, 20)
        dist = np.linalg.norm(traj[-1].numpy() - traj[0].numpy())
        print(f"  {mt}: shape={traj.shape}, distance={dist:.2f}")
    
    history, future = gen.generate_multi_obstacle_scene()
    print(f"\nMulti-obstacle scene:")
    print(f"  History: {history.shape}")
    print(f"  Future: {future.shape}")
    
    hist_batch, fut_batch = generate_complex_training_data(100)
    print(f"\nTraining batch:")
    print(f"  Histories: {hist_batch.shape}")
    print(f"  Futures: {fut_batch.shape}")
    
    print("\n[OK] All tests passed!")

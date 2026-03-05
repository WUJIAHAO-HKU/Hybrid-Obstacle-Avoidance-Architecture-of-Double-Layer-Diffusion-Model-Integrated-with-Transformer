"""
可训练的障碍物预测扩散模型

上层扩散模型需要训练！
- 输入: 障碍物历史观测序列
- 输出: 预测未来位置分布
- 训练数据: 障碍物运动轨迹数据集

Author: Dynamic MPD Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """扩散模型配置"""
    obs_history_len: int = 8       # 观测历史长度
    pred_horizon: int = 12         # 预测步数
    hidden_dim: int = 128          # 隐藏层维度
    num_layers: int = 3            # 网络层数
    diffusion_steps: int = 50      # 扩散步数
    learning_rate: float = 1e-3
    batch_size: int = 32


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码（用于时间步嵌入）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ObstacleDenoiseNet(nn.Module):
    """
    障碍物轨迹去噪网络
    
    给定噪声轨迹和历史观测，预测去噪后的轨迹
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # 历史编码器 (处理观测历史)
        self.history_encoder = nn.Sequential(
            nn.Linear(config.obs_history_len * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 噪声轨迹编码器
        self.traj_encoder = nn.Sequential(
            nn.Linear(config.pred_horizon * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 融合 + 去噪
        self.denoise_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            ) for _ in range(config.num_layers)
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU()
        )
        
        # 输出层 (预测噪声)
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.pred_horizon * 2)
        )
    
    def forward(
        self,
        noisy_traj: torch.Tensor,    # [B, pred_horizon, 2]
        history: torch.Tensor,        # [B, obs_history_len, 2]
        timestep: torch.Tensor        # [B]
    ) -> torch.Tensor:
        """预测噪声"""
        B = noisy_traj.shape[0]
        
        # 编码
        t_emb = self.time_embed(timestep)                              # [B, H]
        h_emb = self.history_encoder(history.reshape(B, -1))           # [B, H]
        traj_emb = self.traj_encoder(noisy_traj.reshape(B, -1))        # [B, H]
        
        # 融合
        x = torch.cat([t_emb, h_emb, traj_emb], dim=-1)                # [B, 3H]
        x = self.fusion(x)                                              # [B, H]
        
        # 多层处理
        for layer in self.denoise_layers:
            x = layer(x) + x  # 残差连接
        
        # 输出噪声预测
        noise_pred = self.output_layer(x)                              # [B, pred*2]
        noise_pred = noise_pred.view(B, self.config.pred_horizon, 2)
        
        return noise_pred


class TrainableObstacleDiffusion(nn.Module):
    """
    可训练的障碍物预测扩散模型
    
    训练流程:
    1. 收集障碍物运动数据 (历史→未来)
    2. 训练去噪网络预测噪声
    3. 推理时从噪声采样，逐步去噪得到预测
    """
    
    def __init__(self, config: DiffusionConfig = None, device='cpu'):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.device = device
        
        # 去噪网络
        self.denoise_net = ObstacleDenoiseNet(self.config).to(device)
        
        # 扩散参数
        self.num_steps = self.config.diffusion_steps
        self.betas = torch.linspace(1e-4, 0.02, self.num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.denoise_net.parameters(), 
            lr=self.config.learning_rate
        )
    
    def q_sample(
        self,
        x0: torch.Tensor,      # 真实轨迹 [B, T, 2]
        t: torch.Tensor        # 时间步 [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向扩散：给真实轨迹添加噪声"""
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t][:, None, None]  # [B, 1, 1]
        
        noisy = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise
        return noisy, noise
    
    def train_step(
        self,
        history: torch.Tensor,     # [B, obs_len, 2]
        future: torch.Tensor       # [B, pred_len, 2]
    ) -> float:
        """训练一步"""
        self.denoise_net.train()
        B = history.shape[0]
        
        # 随机时间步
        t = torch.randint(0, self.num_steps, (B,), device=self.device)
        
        # 添加噪声
        noisy_future, noise = self.q_sample(future, t)
        
        # 预测噪声
        noise_pred = self.denoise_net(noisy_future, history, t)
        
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def predict(
        self,
        history: torch.Tensor,     # [obs_len, num_obs, 2] or [B, obs_len, 2]
        num_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        预测未来轨迹分布
        
        Returns:
            mean: 均值轨迹 [pred_len, 2] or [pred_len, num_obs, 2]
            std: 标准差 [pred_len, 2] or [pred_len, num_obs, 2]
            confidence: 置信度
        """
        self.denoise_net.eval()
        
        # 处理输入形状
        if history.dim() == 3 and history.shape[1] > 1:
            # 多障碍物情况 [obs_len, num_obs, 2]
            return self._predict_multi_obstacle(history, num_samples)
        
        # 单障碍物 [obs_len, 2] -> [1, obs_len, 2]
        if history.dim() == 2:
            history = history.unsqueeze(0)
        
        pred_len = self.config.pred_horizon
        all_samples = []
        
        for _ in range(num_samples):
            # 从噪声开始
            x = torch.randn(1, pred_len, 2, device=self.device)
            
            # 逐步去噪
            for step in reversed(range(self.num_steps)):
                t = torch.tensor([step], device=self.device)
                
                # 预测噪声
                noise_pred = self.denoise_net(x, history, t)
                
                # 去噪
                alpha = self.alphas[step]
                alpha_bar = self.alpha_bars[step]
                
                x = (1 / alpha.sqrt()) * (
                    x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred
                )
                
                # 添加噪声（除了最后一步）
                if step > 0:
                    noise = torch.randn_like(x) * self.betas[step].sqrt()
                    x = x + noise
            
            all_samples.append(x.squeeze(0))
        
        # 统计
        samples = torch.stack(all_samples)  # [num_samples, pred_len, 2]
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        confidence = 1.0 / (1.0 + std.mean().item() * 10)
        
        return mean, std, confidence
    
    def _predict_multi_obstacle(
        self,
        history: torch.Tensor,     # [obs_len, num_obs, 2]
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """预测多个障碍物"""
        obs_len, num_obs, _ = history.shape
        pred_len = self.config.pred_horizon
        
        all_means = []
        all_stds = []
        all_confs = []
        
        for obs_idx in range(num_obs):
            obs_history = history[:, obs_idx, :]  # [obs_len, 2]
            mean, std, conf = self.predict(obs_history, num_samples)
            all_means.append(mean)
            all_stds.append(std)
            all_confs.append(conf)
        
        means = torch.stack(all_means, dim=1)  # [pred_len, num_obs, 2]
        stds = torch.stack(all_stds, dim=1)
        confidence = np.mean(all_confs)
        
        return means, stds, confidence
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'config': self.config,
            'state_dict': self.denoise_net.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.denoise_net.load_state_dict(checkpoint['state_dict'])


def generate_training_data(
    num_trajectories: int = 1000,
    obs_len: int = 8,
    pred_len: int = 12,
    device='cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成训练数据（模拟障碍物运动）
    
    运动模式:
    1. 匀速直线
    2. 匀速转弯
    3. 加速/减速
    4. 随机扰动
    """
    histories = []
    futures = []
    
    for _ in range(num_trajectories):
        # 随机初始位置和速度
        pos = torch.randn(2) * 0.5
        vel = torch.randn(2) * 0.03
        
        # 随机运动模式
        mode = np.random.choice(['linear', 'turn', 'accel', 'random'])
        
        traj = [pos.clone()]
        
        for t in range(obs_len + pred_len - 1):
            if mode == 'linear':
                pos = pos + vel
            elif mode == 'turn':
                angle = 0.1 * np.sin(t * 0.5)
                rot = torch.tensor([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]], dtype=torch.float32)
                vel = rot @ vel
                pos = pos + vel
            elif mode == 'accel':
                vel = vel * (1 + 0.02 * np.sin(t * 0.3))
                pos = pos + vel
            else:  # random
                pos = pos + vel + torch.randn(2) * 0.01
            
            traj.append(pos.clone())
        
        traj = torch.stack(traj)
        histories.append(traj[:obs_len])
        futures.append(traj[obs_len:])
    
    return torch.stack(histories).to(device), torch.stack(futures).to(device)


def train_obstacle_diffusion(
    num_epochs: int = 500,
    num_data: int = 2000,
    device='cpu'
) -> TrainableObstacleDiffusion:
    """训练障碍物预测扩散模型"""
    
    print("=" * 50)
    print("训练障碍物预测扩散模型")
    print("=" * 50)
    
    config = DiffusionConfig()
    model = TrainableObstacleDiffusion(config, device)
    
    # 生成训练数据
    print(f"\n生成 {num_data} 条训练轨迹...")
    histories, futures = generate_training_data(num_data, device=device)
    
    # 训练
    print(f"\n开始训练 ({num_epochs} epochs)...")
    batch_size = config.batch_size
    
    for epoch in range(num_epochs):
        # 打乱数据
        perm = torch.randperm(num_data)
        total_loss = 0
        num_batches = 0
        
        for i in range(0, num_data, batch_size):
            idx = perm[i:i+batch_size]
            loss = model.train_step(histories[idx], futures[idx])
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}")
    
    print("\n训练完成!")
    return model

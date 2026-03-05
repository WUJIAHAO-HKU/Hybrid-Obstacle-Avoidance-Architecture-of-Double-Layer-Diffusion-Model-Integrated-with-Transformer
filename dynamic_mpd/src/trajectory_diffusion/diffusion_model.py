"""
障碍物轨迹预测扩散模型

基于DDPM的条件扩散模型，用于预测动态障碍物（行人等）的未来轨迹。

架构:
    1. 历史轨迹 -> Transformer编码 -> 上下文特征
    2. 上下文特征 + 噪声 -> UNet/Transformer -> 预测噪声
    3. 反向扩散过程 -> 生成未来轨迹

参考:
- MID: Motion Indeterminacy Diffusion (Gu et al., CVPR 2022)
- Diffusion-based Trajectory Prediction

Author: Dynamic MPD Project
Date: 2026-01-23
"""

import math
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_encoder import (
    TrajectoryTransformer,
    SocialTransformer,
    TimeEmbedding,
)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    余弦噪声调度
    
    相比线性调度，在高信噪比区域更平滑。
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """线性噪声调度"""
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionDenoiser(nn.Module):
    """
    扩散去噪网络
    
    预测添加到轨迹上的噪声。
    
    架构:
        [noisy_trajectory, context, time_emb] -> Transformer/MLP -> noise_pred
    """
    
    def __init__(
        self,
        traj_dim: int = 2,
        pred_len: int = 12,
        context_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.traj_dim = traj_dim
        self.pred_len = pred_len
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # 时间嵌入
        self.time_emb = TimeEmbedding(hidden_dim)
        
        # 轨迹嵌入
        self.traj_embed = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 上下文融合
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, pred_len, hidden_dim) * 0.02)
        
        # Transformer去噪器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, traj_dim),
        )
        
    def forward(
        self,
        noisy_traj: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测噪声
        
        Args:
            noisy_traj: [batch, pred_len, traj_dim] 带噪声的轨迹
            t: [batch] 扩散时间步
            context: [batch, context_dim] 上下文特征
            
        Returns:
            noise_pred: [batch, pred_len, traj_dim] 预测的噪声
        """
        batch_size = noisy_traj.shape[0]
        
        # 时间嵌入
        t_emb = self.time_emb(t)  # [batch, hidden_dim]
        
        # 轨迹嵌入
        traj_emb = self.traj_embed(noisy_traj)  # [batch, pred_len, hidden_dim]
        traj_emb = traj_emb + self.pos_embed  # 加位置编码
        
        # 上下文处理
        ctx = self.context_proj(context)  # [batch, hidden_dim]
        
        # 融合时间和上下文作为memory
        memory = (ctx + t_emb).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Transformer解码
        decoded = self.transformer(traj_emb, memory)
        
        # 输出投影
        noise_pred = self.output_proj(decoded)
        
        return noise_pred


class ObstacleTrajectoryDiffusion(nn.Module):
    """
    障碍物轨迹预测扩散模型
    
    完整的条件扩散模型，包括:
    - 历史轨迹编码器
    - 社交交互建模
    - DDPM扩散过程
    - 多样本采样
    """
    
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        traj_dim: int = 2,
        d_model: int = 128,
        denoiser_hidden: int = 256,
        encoder_layers: int = 4,
        denoiser_layers: int = 4,
        nhead: int = 8,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "cosine",
        dropout: float = 0.1,
        use_social: bool = True,
    ):
        """
        Args:
            obs_len: 观测长度 (历史帧数)
            pred_len: 预测长度 (未来帧数)
            traj_dim: 轨迹维度 (2D: x, y)
            d_model: Transformer编码器维度
            denoiser_hidden: 去噪器隐藏维度
            encoder_layers: 编码器层数
            denoiser_layers: 去噪器层数
            nhead: 注意力头数
            num_diffusion_steps: 扩散步数
            beta_schedule: 噪声调度 ('linear' or 'cosine')
            dropout: Dropout率
            use_social: 是否使用社交交互建模
        """
        super().__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_dim = traj_dim
        self.d_model = d_model
        self.num_diffusion_steps = num_diffusion_steps
        self.use_social = use_social
        
        # 历史轨迹编码器
        self.encoder = TrajectoryTransformer(
            input_dim=traj_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=encoder_layers,
            dropout=dropout,
        )
        
        # 社交交互模块
        if use_social:
            self.social = SocialTransformer(
                d_model=d_model,
                nhead=nhead,
                num_layers=2,
                dropout=dropout,
            )
        else:
            self.social = None
            
        # 去噪网络
        self.denoiser = DiffusionDenoiser(
            traj_dim=traj_dim,
            pred_len=pred_len,
            context_dim=d_model,
            hidden_dim=denoiser_hidden,
            num_layers=denoiser_layers,
            nhead=nhead,
            dropout=dropout,
        )
        
        # 设置扩散参数
        self._setup_diffusion(beta_schedule)
        
    def _setup_diffusion(self, schedule: str):
        """设置扩散过程参数"""
        if schedule == "cosine":
            betas = cosine_beta_schedule(self.num_diffusion_steps)
        else:
            betas = linear_beta_schedule(self.num_diffusion_steps)
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为buffer (不参与梯度计算，但会保存到state_dict)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # 后验方差
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向扩散过程: q(x_t | x_0)
        
        Args:
            x_start: [batch, pred_len, traj_dim] 干净数据
            t: [batch] 时间步
            noise: 可选的噪声
            
        Returns:
            x_t: 加噪后的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算扩散损失
        
        Args:
            x_start: [batch, pred_len, traj_dim] 真实未来轨迹
            context: [batch, d_model] 上下文特征
            t: [batch] 时间步
            
        Returns:
            loss: MSE损失
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 前向加噪
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        noise_pred = self.denoiser(x_noisy, t.float(), context)
        
        # MSE损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        neighbors_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        训练前向传播
        
        Args:
            history: [batch, obs_len, traj_dim] 自身历史轨迹
            future: [batch, pred_len, traj_dim] 真实未来轨迹
            neighbors_history: [batch, num_neighbors, obs_len, traj_dim] 邻居历史
            
        Returns:
            loss: 扩散损失
        """
        batch_size = history.shape[0]
        device = history.device
        
        # 编码历史轨迹
        context = self.encoder(history)  # [batch, d_model]
        
        # 社交交互
        if self.use_social and neighbors_history is not None:
            # 编码所有邻居
            all_histories = torch.cat([
                history.unsqueeze(1),  # [batch, 1, obs_len, traj_dim]
                neighbors_history
            ], dim=1)
            
            all_contexts = self.encoder.encode_batch(all_histories)
            social_contexts = self.social(all_contexts)
            context = social_contexts[:, 0, :]  # 取自身的社交感知特征
        
        # 随机采样时间步
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        
        # 计算损失
        loss = self.p_losses(future, context, t)
        
        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        反向扩散单步: p(x_{t-1} | x_t)
        """
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        # 预测噪声
        noise_pred = self.denoiser(x, t_tensor.float(), context)
        
        # 计算均值
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        beta = self.betas[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        model_mean = sqrt_recip_alpha * (
            x - beta * noise_pred / sqrt_one_minus_alpha_cumprod
        )
        
        if t > 0:
            noise = torch.randn_like(x)
            posterior_variance = self.posterior_variance[t]
            return model_mean + torch.sqrt(posterior_variance) * noise
        else:
            return model_mean
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        context: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        完整的反向扩散采样
        
        Args:
            context: [batch, d_model] 上下文特征
            n_samples: 每个上下文生成的样本数
            
        Returns:
            samples: [batch * n_samples, pred_len, traj_dim] 生成的轨迹
        """
        batch_size = context.shape[0]
        device = context.device
        
        # 扩展context以生成多个样本
        if n_samples > 1:
            context = context.repeat_interleave(n_samples, dim=0)
        
        # 从噪声开始
        shape = (batch_size * n_samples, self.pred_len, self.traj_dim)
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for t in reversed(range(self.num_diffusion_steps)):
            x = self.p_sample(x, t, context)
            
        return x
    
    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        neighbors_history: Optional[torch.Tensor] = None,
        n_samples: int = 20,
    ) -> torch.Tensor:
        """
        从历史轨迹采样未来轨迹
        
        Args:
            history: [batch, obs_len, traj_dim] 历史轨迹
            neighbors_history: [batch, num_neighbors, obs_len, traj_dim] 邻居历史
            n_samples: 每个场景的采样数量
            
        Returns:
            samples: [batch, n_samples, pred_len, traj_dim] 预测轨迹
        """
        batch_size = history.shape[0]
        
        # 编码
        context = self.encoder(history)
        
        # 社交交互
        if self.use_social and neighbors_history is not None:
            all_histories = torch.cat([
                history.unsqueeze(1),
                neighbors_history
            ], dim=1)
            all_contexts = self.encoder.encode_batch(all_histories)
            social_contexts = self.social(all_contexts)
            context = social_contexts[:, 0, :]
        
        # 采样
        samples = self.p_sample_loop(context, n_samples)
        
        # 重新排列形状
        samples = samples.view(batch_size, n_samples, self.pred_len, self.traj_dim)
        
        return samples


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Obstacle Trajectory Diffusion 测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模型参数
    obs_len = 8
    pred_len = 12
    traj_dim = 2
    batch_size = 4
    
    # 创建模型
    model = ObstacleTrajectoryDiffusion(
        obs_len=obs_len,
        pred_len=pred_len,
        traj_dim=traj_dim,
        d_model=128,
        denoiser_hidden=256,
        num_diffusion_steps=100,
        use_social=True,
    ).to(device)
    
    # 测试数据
    history = torch.randn(batch_size, obs_len, traj_dim, device=device)
    future = torch.randn(batch_size, pred_len, traj_dim, device=device)
    neighbors = torch.randn(batch_size, 5, obs_len, traj_dim, device=device)
    
    print(f"\n输入:")
    print(f"  history: {history.shape}")
    print(f"  future: {future.shape}")
    print(f"  neighbors: {neighbors.shape}")
    
    # 训练前向传播
    loss = model(history, future, neighbors)
    print(f"\n训练损失: {loss.item():.4f}")
    
    # 采样测试
    print(f"\n采样测试 (n_samples=5)...")
    samples = model.sample(history, neighbors, n_samples=5)
    print(f"  采样输出: {samples.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

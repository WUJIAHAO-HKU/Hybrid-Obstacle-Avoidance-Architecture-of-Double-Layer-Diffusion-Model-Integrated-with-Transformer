"""
Transformer-based Obstacle Trajectory Diffusion Model

创新点:
1. 用Temporal Transformer替代MLP，捕获时序依赖
2. Self-Attention建模障碍物历史轨迹的长程依赖
3. Cross-Attention建模多障碍物之间的交互关系
4. 与原有DDPM框架完全兼容

适用顶会: CoRL, ICRA, IROS, RAL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class TransformerDiffusionConfig:
    """Transformer扩散模型配置"""
    # Transformer参数
    d_model: int = 128           # 模型维度
    n_heads: int = 4             # 注意力头数
    n_layers: int = 4            # Transformer层数
    d_ff: int = 512              # FFN隐藏维度
    dropout: float = 0.1
    
    # 扩散参数
    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # 任务参数
    obs_history_len: int = 8     # 观测历史长度
    pred_horizon: int = 12       # 预测时长
    state_dim: int = 2           # 状态维度 (x, y)
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 64
    
    # 多障碍物交互
    max_num_obstacles: int = 8   # 最大障碍物数量
    use_cross_attention: bool = True  # 是否使用跨障碍物注意力


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码 (用于时间步和序列位置)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(out)


class CrossAttention(nn.Module):
    """跨障碍物注意力 - 建模障碍物之间的交互"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, q_len, _ = query.shape
        _, c_len, _ = context.shape
        
        Q = self.W_q(query).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context).view(batch_size, c_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context).view(batch_size, c_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """Transformer块 with optional cross-attention"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, use_cross_attn: bool = False):
        super().__init__()
        
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttention(d_model, n_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention (if enabled and context provided)
        if self.use_cross_attn and context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


class AdaptiveLayerNorm(nn.Module):
    """自适应层归一化 - 用于注入时间步信息"""
    
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * d_model)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [batch, cond_dim] -> [batch, 1, 2*d_model]
        scale_shift = self.proj(cond).unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TemporalTransformerEncoder(nn.Module):
    """时序Transformer编码器 - 处理历史轨迹"""
    
    def __init__(self, config: TransformerDiffusionConfig):
        super().__init__()
        self.config = config
        
        # 输入嵌入
        self.input_proj = nn.Linear(config.state_dim, config.d_model)
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.obs_history_len, config.d_model) * 0.02
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, 
                config.dropout, use_cross_attn=False
            )
            for _ in range(config.n_layers // 2)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: [batch, obs_history_len, state_dim]
        Returns:
            encoded: [batch, obs_history_len, d_model]
        """
        x = self.input_proj(history)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)


class DiffusionTransformerDecoder(nn.Module):
    """扩散Transformer解码器 - 生成预测轨迹"""
    
    def __init__(self, config: TransformerDiffusionConfig):
        super().__init__()
        self.config = config
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(config.d_model),
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )
        
        # 输入嵌入 (noisy prediction)
        self.input_proj = nn.Linear(config.state_dim, config.d_model)
        
        # 预测位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.pred_horizon, config.d_model) * 0.02
        )
        
        # 自适应LayerNorm (注入时间步)
        self.ada_norms = nn.ModuleList([
            AdaptiveLayerNorm(config.d_model, config.d_model)
            for _ in range(config.n_layers)
        ])
        
        # Transformer层 (with cross-attention to history)
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.dropout, use_cross_attn=True
            )
            for _ in range(config.n_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(config.d_model, config.state_dim)
        
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: noisy trajectory [batch, pred_horizon, state_dim]
            t: diffusion timestep [batch]
            context: encoded history [batch, obs_history_len, d_model]
        Returns:
            noise_pred: [batch, pred_horizon, state_dim]
        """
        # 时间嵌入
        t_emb = self.time_embed(t.float())
        
        # 输入嵌入
        x = self.input_proj(x_t)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Transformer layers with time conditioning
        for i, layer in enumerate(self.layers):
            x = self.ada_norms[i](x, t_emb)
            x = layer(x, context)
        
        return self.output_proj(x)


class TransformerObstacleDiffusion(nn.Module):
    """
    Transformer-based 障碍物轨迹扩散模型
    
    创新点:
    1. Temporal Transformer编码历史轨迹
    2. Cross-Attention建模历史-预测依赖
    3. Adaptive LayerNorm注入扩散时间步
    4. 支持多障碍物交互 (可选)
    """
    
    def __init__(self, config: Optional[TransformerDiffusionConfig] = None):
        super().__init__()
        self.config = config or TransformerDiffusionConfig()
        
        # 编码器 (历史轨迹)
        self.encoder = TemporalTransformerEncoder(self.config)
        
        # 解码器 (扩散生成)
        self.decoder = DiffusionTransformerDecoder(self.config)
        
        # 扩散参数
        self._setup_diffusion()
        
        # 多障碍物交互 (可选)
        if self.config.use_cross_attention:
            self.obstacle_interaction = CrossAttention(
                self.config.d_model, self.config.n_heads, self.config.dropout
            )
        
    def _setup_diffusion(self):
        """设置扩散过程参数"""
        steps = self.config.diffusion_steps
        
        # Beta schedule (linear)
        betas = torch.linspace(
            self.config.beta_start, self.config.beta_end, steps
        )
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # Posterior variance
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def forward(self, history: torch.Tensor, future: torch.Tensor,
                other_obstacles: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        训练前向传播
        
        Args:
            history: [batch, obs_history_len, state_dim] 障碍物历史轨迹
            future: [batch, pred_horizon, state_dim] 真实未来轨迹
            other_obstacles: [batch, num_other, obs_history_len, state_dim] 其他障碍物 (可选)
        
        Returns:
            loss: 扩散损失
        """
        batch_size = history.shape[0]
        device = history.device
        
        # 编码历史
        context = self.encoder(history)
        
        # 多障碍物交互 (如果有其他障碍物)
        if other_obstacles is not None and self.config.use_cross_attention:
            # 编码其他障碍物
            B, N, T, D = other_obstacles.shape
            other_encoded = self.encoder(other_obstacles.view(B*N, T, D))
            other_encoded = other_encoded.view(B, N, T, -1)
            
            # 聚合其他障碍物信息
            other_context = other_encoded.mean(dim=2)  # [B, N, d_model]
            context_pooled = context.mean(dim=1, keepdim=True)  # [B, 1, d_model]
            
            # Cross-attention
            interaction = self.obstacle_interaction(context_pooled, other_context)
            context = context + interaction.expand_as(context)
        
        # 随机时间步
        t = torch.randint(0, self.config.diffusion_steps, (batch_size,), device=device)
        
        # 添加噪声
        noise = torch.randn_like(future)
        x_t = self.q_sample(future, t, noise)
        
        # 预测噪声
        noise_pred = self.decoder(x_t, t, context)
        
        # MSE损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def predict(self, history: torch.Tensor, num_samples: int = 1,
                other_obstacles: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        预测未来轨迹
        
        Args:
            history: [obs_history_len, state_dim] 单个障碍物历史
            num_samples: 采样数量
            other_obstacles: [num_other, obs_history_len, state_dim] 其他障碍物
        
        Returns:
            mean: [pred_horizon, state_dim] 预测均值
            std: [pred_horizon, state_dim] 预测标准差
            samples: 所有采样轨迹
        """
        device = next(self.parameters()).device
        
        # 扩展batch维度
        history = history.unsqueeze(0).expand(num_samples, -1, -1).to(device)
        
        # 编码历史
        context = self.encoder(history)
        
        # 多障碍物交互
        if other_obstacles is not None and self.config.use_cross_attention:
            other_obstacles = other_obstacles.unsqueeze(0).expand(num_samples, -1, -1, -1).to(device)
            B, N, T, D = other_obstacles.shape
            other_encoded = self.encoder(other_obstacles.view(B*N, T, D))
            other_encoded = other_encoded.view(B, N, T, -1)
            
            other_context = other_encoded.mean(dim=2)
            context_pooled = context.mean(dim=1, keepdim=True)
            interaction = self.obstacle_interaction(context_pooled, other_context)
            context = context + interaction.expand_as(context)
        
        # 从噪声开始
        x = torch.randn(num_samples, self.config.pred_horizon, self.config.state_dim, device=device)
        
        # 扩散采样
        for t in reversed(range(self.config.diffusion_steps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.decoder(x, t_batch, context)
            
            # DDPM更新
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # 预测x_0
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
            
            # 计算均值
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]
            mean = (torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * x0_pred +
                    torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * x)
            
            if t > 0:
                variance = self.posterior_variance[t]
                x = mean + torch.sqrt(variance) * torch.randn_like(x)
            else:
                x = mean
        
        samples = [x[i].cpu() for i in range(num_samples)]
        mean = x.mean(dim=0).cpu()
        std = x.std(dim=0).cpu()
        
        return mean, std, samples


class TransformerDiffusionTrainer:
    """训练器"""
    
    def __init__(self, model: TransformerObstacleDiffusion, 
                 config: TransformerDiffusionConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
    def train_step(self, history: torch.Tensor, future: torch.Tensor,
                   other_obstacles: Optional[torch.Tensor] = None) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.model(history, future, other_obstacles)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()


# ============ 与原有接口兼容的包装类 ============

class TransformerObstacleDiffusionWrapper:
    """
    包装类 - 与原有 TrainableObstacleDiffusion 接口完全兼容
    可直接替换使用
    
    关键改进: 使用相对位移归一化，避免模型学习到"平均位置"
    - 训练时: 将历史和未来轨迹都转换为相对于历史末端点的位移
    - 预测时: 将预测的相对位移转换回绝对坐标
    """
    
    def __init__(self, config: Optional[TransformerDiffusionConfig] = None, device='cuda'):
        self.config = config or TransformerDiffusionConfig()
        self.device = device
        self.model = TransformerObstacleDiffusion(self.config).to(device)
        self.trainer = TransformerDiffusionTrainer(self.model, self.config)
        self.training_losses = []
    
    def _normalize_to_relative(self, history: torch.Tensor, future: torch.Tensor):
        """
        将绝对坐标转换为相对于历史末端点的位移
        
        Args:
            history: [batch, hist_len, 2] 历史轨迹
            future: [batch, pred_len, 2] 未来轨迹
        
        Returns:
            rel_history: [batch, hist_len, 2] 相对历史（以末端为原点）
            rel_future: [batch, pred_len, 2] 相对未来
            anchor: [batch, 2] 锚点（历史末端点）
        """
        anchor = history[:, -1:, :]  # [batch, 1, 2]
        rel_history = history - anchor
        rel_future = future - anchor
        return rel_history, rel_future, anchor.squeeze(1)
    
    def _denormalize_from_relative(self, rel_future: torch.Tensor, anchor: torch.Tensor):
        """
        将相对位移转换回绝对坐标
        
        Args:
            rel_future: [batch, pred_len, 2] 相对预测
            anchor: [batch, 2] 锚点
        
        Returns:
            abs_future: [batch, pred_len, 2] 绝对坐标预测
        """
        return rel_future + anchor.unsqueeze(1)
        
    def train_epoch(self, history_batch: torch.Tensor, future_batch: torch.Tensor) -> float:
        """训练一个epoch - 使用相对位移"""
        history_batch = history_batch.to(self.device)
        future_batch = future_batch.to(self.device)
        
        # 转换为相对坐标
        rel_history, rel_future, _ = self._normalize_to_relative(history_batch, future_batch)
        
        batch_size = self.config.batch_size
        n_samples = rel_history.shape[0]
        total_loss = 0
        n_batches = 0
        
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            history = rel_history[batch_idx]
            future = rel_future[batch_idx]
            
            loss = self.trainer.train_step(history, future)
            total_loss += loss
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        self.training_losses.append(avg_loss)
        return avg_loss
    
    def predict(self, history: torch.Tensor, num_samples: int = 12):
        """预测 - 使用相对位移并转换回绝对坐标"""
        self.model.eval()
        history = history.to(self.device)
        
        # 获取锚点（历史末端）
        if history.dim() == 2:
            anchor = history[-1:, :]  # [1, 2]
            rel_history = history - anchor
        else:
            anchor = history[:, -1, :]  # [batch, 2]
            rel_history = history - anchor.unsqueeze(1)
        
        # 模型预测相对位移
        rel_mean, rel_std, rel_samples = self.model.predict(rel_history, num_samples)
        
        # 转换回绝对坐标
        if anchor.dim() == 2 and anchor.shape[0] == 1:
            anchor_2d = anchor.squeeze(0)  # [2]
        else:
            anchor_2d = anchor
        
        abs_mean = rel_mean + anchor_2d.cpu()
        abs_samples = [s + anchor_2d.cpu() for s in rel_samples]
        
        return abs_mean, rel_std, abs_samples
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_losses': self.training_losses
        }, path)
        print(f"  Model saved to: {path}")
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'training_losses' in checkpoint:
            self.training_losses = checkpoint['training_losses']
        print(f"  Model loaded from: {path}")
    
    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = 'cuda'):
        """
        从checkpoint加载模型（静态方法，用于正式仿真）
        
        Args:
            path: checkpoint文件路径
            device: 运行设备
            
        Returns:
            TransformerObstacleDiffusionWrapper: 加载好的模型
            
        Usage:
            model = TransformerObstacleDiffusionWrapper.load_from_checkpoint(
                'trained_models/best_transformer_obstacle_model.pt',
                device='cuda'
            )
            prediction = model.predict(obstacle_history)
        """
        # PyTorch 2.6+ 需要 weights_only=False 或添加 safe_globals
        # 这里使用 weights_only=False 因为我们信任自己训练的模型
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        wrapper = cls(config=config, device=device)
        wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        if 'training_losses' in checkpoint:
            wrapper.training_losses = checkpoint['training_losses']
        
        wrapper.model.eval()
        print(f"  [OK] Transformer model loaded from: {path}")
        print(f"       Config: d_model={config.d_model}, n_layers={config.n_layers}")
        print(f"       Training epochs: {len(wrapper.training_losses)}")
        
        return wrapper


# ============ 测试代码 ============

if __name__ == '__main__':
    print("=" * 60)
    print("  Transformer-based Obstacle Diffusion Model")
    print("  创新点: Temporal Attention + Cross-Attention")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    # 创建模型
    config = TransformerDiffusionConfig(
        d_model=128,
        n_heads=4,
        n_layers=4,
        diffusion_steps=50,
        obs_history_len=8,
        pred_horizon=12
    )
    
    model = TransformerObstacleDiffusionWrapper(config, device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # 生成测试数据
    print("\n  Generating test data...")
    n_samples = 500
    history = torch.randn(n_samples, config.obs_history_len, config.state_dim)
    future = torch.randn(n_samples, config.pred_horizon, config.state_dim)
    
    # 训练测试
    print("\n  Training test (5 epochs)...")
    for epoch in range(5):
        loss = model.train_epoch(history, future)
        print(f"    Epoch {epoch+1}: Loss = {loss:.4f}")
    
    # 预测测试
    print("\n  Prediction test...")
    test_history = torch.randn(config.obs_history_len, config.state_dim)
    mean, std, samples = model.predict(test_history, num_samples=10)
    print(f"    Prediction shape: {mean.shape}")
    print(f"    Mean uncertainty: {std.mean().item():.4f}")
    
    print("\n  ✓ Transformer Diffusion Model Ready!")
    print("=" * 60)

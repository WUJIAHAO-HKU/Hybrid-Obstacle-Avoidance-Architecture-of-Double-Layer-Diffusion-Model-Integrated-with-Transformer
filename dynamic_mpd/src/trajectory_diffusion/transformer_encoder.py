"""
Transformer编码器模块

用于编码历史轨迹序列，提取时空特征。

参考:
- Trajectron++ (Salzmann et al., ECCV 2020)
- MID (Gu et al., CVPR 2022) - Motion Indeterminacy Diffusion

Author: Dynamic MPD Project
Date: 2026-01-23
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    为序列中的每个位置添加唯一的位置信息。
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    """
    扩散时间步嵌入
    
    使用正弦编码将扩散时间步转换为特征向量。
    """
    
    def __init__(self, dim: int, max_timesteps: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_timesteps = max_timesteps
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch] 扩散时间步
        Returns:
            [batch, dim] 时间嵌入
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class TrajectoryTransformer(nn.Module):
    """
    轨迹Transformer编码器
    
    将历史轨迹序列编码为上下文特征，用于条件扩散生成。
    
    架构:
        Input: [batch, obs_len, 2] (历史轨迹坐标)
        -> Input Projection
        -> Positional Encoding
        -> Transformer Encoder Layers
        -> Context Feature [batch, d_model]
    """
    
    def __init__(
        self,
        input_dim: int = 2,  # x, y坐标
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ):
        """
        Args:
            input_dim: 输入维度 (2 for 2D coordinates)
            d_model: Transformer隐藏维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # 输出投影 (将序列聚合为单个上下文向量)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # CLS token用于序列聚合
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(
        self,
        history: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码历史轨迹
        
        Args:
            history: [batch, obs_len, input_dim] 历史轨迹
            src_key_padding_mask: [batch, obs_len] padding mask (True表示padding)
            
        Returns:
            context: [batch, d_model] 上下文特征
        """
        batch_size = history.shape[0]
        
        # 输入投影
        x = self.input_projection(history)  # [batch, obs_len, d_model]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 1+obs_len, d_model]
        
        # 更新padding mask
        if src_key_padding_mask is not None:
            # 为CLS token添加False (不mask)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=history.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 取CLS token作为序列表示
        context = x[:, 0, :]  # [batch, d_model]
        
        # 输出投影
        context = self.output_projection(context)
        
        return context
    
    def encode_batch(
        self,
        histories: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        批量编码多个智能体的历史轨迹
        
        Args:
            histories: [batch, num_agents, obs_len, 2] 多智能体历史
            lengths: [batch, num_agents] 每个智能体的有效长度
            
        Returns:
            contexts: [batch, num_agents, d_model]
        """
        batch_size, num_agents, obs_len, _ = histories.shape
        
        # 展平batch和agent维度
        histories_flat = histories.view(-1, obs_len, self.input_dim)
        
        # 创建padding mask
        if lengths is not None:
            lengths_flat = lengths.view(-1)
            max_len = obs_len
            indices = torch.arange(max_len, device=histories.device).expand(
                batch_size * num_agents, -1
            )
            mask = indices >= lengths_flat.unsqueeze(1)
        else:
            mask = None
        
        # 编码
        contexts_flat = self.forward(histories_flat, mask)
        
        # 恢复形状
        contexts = contexts_flat.view(batch_size, num_agents, self.d_model)
        
        return contexts


class SocialTransformer(nn.Module):
    """
    社交Transformer
    
    建模多个智能体之间的交互关系。
    
    架构:
        Input: [batch, num_agents, d_model] (每个智能体的特征)
        -> Self-Attention across agents
        -> Cross-agent interaction modeling
        -> Output: [batch, num_agents, d_model]
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 社交注意力层
        social_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.social_transformer = nn.TransformerEncoder(
            social_layer,
            num_layers=num_layers,
        )
        
    def forward(
        self,
        agent_features: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        建模智能体间交互
        
        Args:
            agent_features: [batch, num_agents, d_model]
            agent_mask: [batch, num_agents] True表示该智能体无效
            
        Returns:
            social_features: [batch, num_agents, d_model]
        """
        return self.social_transformer(agent_features, src_key_padding_mask=agent_mask)


class TemporalDecoder(nn.Module):
    """
    时序解码器
    
    从上下文特征解码未来轨迹。
    """
    
    def __init__(
        self,
        d_model: int = 128,
        output_dim: int = 2,
        pred_len: int = 12,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # 查询embedding (用于解码未来时间步)
        self.query_embed = nn.Parameter(torch.randn(pred_len, d_model))
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(
        self,
        context: torch.Tensor,
        noisy_future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        解码未来轨迹
        
        Args:
            context: [batch, d_model] 上下文特征
            noisy_future: [batch, pred_len, output_dim] 带噪声的未来轨迹 (扩散训练时)
            
        Returns:
            predicted: [batch, pred_len, output_dim] 预测轨迹
        """
        batch_size = context.shape[0]
        
        # 扩展context为memory
        memory = context.unsqueeze(1)  # [batch, 1, d_model]
        
        # 查询向量
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        if noisy_future is not None:
            # 扩散训练: 融合噪声未来信息
            noisy_embed = nn.Linear(self.output_dim, self.d_model).to(noisy_future.device)
            queries = queries + noisy_embed(noisy_future)
        
        # 解码
        decoded = self.transformer_decoder(queries, memory)
        
        # 输出投影
        output = self.output_projection(decoded)
        
        return output


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Trajectory Transformer 测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试参数
    batch_size = 4
    obs_len = 8
    pred_len = 12
    input_dim = 2
    d_model = 128
    
    # 创建模型
    encoder = TrajectoryTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=8,
        num_encoder_layers=4,
    ).to(device)
    
    social = SocialTransformer(
        d_model=d_model,
        nhead=8,
        num_layers=2,
    ).to(device)
    
    decoder = TemporalDecoder(
        d_model=d_model,
        output_dim=input_dim,
        pred_len=pred_len,
    ).to(device)
    
    # 测试数据
    history = torch.randn(batch_size, obs_len, input_dim, device=device)
    
    print(f"\n输入:")
    print(f"  history: {history.shape}")
    
    # 编码
    context = encoder(history)
    print(f"\n编码器输出:")
    print(f"  context: {context.shape}")
    
    # 多智能体场景
    num_agents = 5
    multi_history = torch.randn(batch_size, num_agents, obs_len, input_dim, device=device)
    multi_context = encoder.encode_batch(multi_history)
    print(f"\n多智能体编码:")
    print(f"  input: {multi_history.shape}")
    print(f"  output: {multi_context.shape}")
    
    # 社交交互
    social_context = social(multi_context)
    print(f"\n社交交互:")
    print(f"  output: {social_context.shape}")
    
    # 解码
    predicted = decoder(context)
    print(f"\n解码器输出:")
    print(f"  predicted: {predicted.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in social.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

"""
Trajectory Diffusion Module

上层扩散模型，用于预测动态障碍物的未来轨迹。

模块组件:
- ObstacleTrajectoryDiffusion: 基于DDPM的轨迹预测模型
- TrajectoryTransformer: Transformer编码器
- SocialTransformer: 社交交互建模
- ETHUCYDataset: ETH/UCY数据集加载器
- SyntheticTrajectoryDataset: 合成轨迹数据集
"""

from .transformer_encoder import (
    TrajectoryTransformer,
    SocialTransformer,
    TemporalDecoder,
    PositionalEncoding,
    TimeEmbedding,
)

from .diffusion_model import (
    ObstacleTrajectoryDiffusion,
    DiffusionDenoiser,
    cosine_beta_schedule,
    linear_beta_schedule,
)

from .dataset import (
    ETHUCYDataset,
    SyntheticTrajectoryDataset,
    create_dataloader,
)

__all__ = [
    # Transformer组件
    'TrajectoryTransformer',
    'SocialTransformer', 
    'TemporalDecoder',
    'PositionalEncoding',
    'TimeEmbedding',
    # 扩散模型
    'ObstacleTrajectoryDiffusion',
    'DiffusionDenoiser',
    'cosine_beta_schedule',
    'linear_beta_schedule',
    # 数据集
    'ETHUCYDataset',
    'SyntheticTrajectoryDataset',
    'create_dataloader',
]

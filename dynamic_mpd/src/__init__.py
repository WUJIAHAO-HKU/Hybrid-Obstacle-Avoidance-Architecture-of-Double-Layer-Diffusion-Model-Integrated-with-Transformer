"""
动态MPD模块初始化
Hierarchical Dual Diffusion for Dynamic Motion Planning

核心模块:
- trainable_obstacle_diffusion: 上层扩散模型 (MLP版)
- transformer_obstacle_diffusion: 上层扩散模型 (Transformer版) ⭐
- complex_obstacle_data: 复杂障碍物数据生成 (11种运动类型)
- mpd_integration: MPD框架集成 (下层扩散)
"""

# 复杂障碍物数据生成 (无外部依赖)
from .complex_obstacle_data import (
    ObstacleMotionConfig,
    ComplexObstacleDataGenerator,
    generate_complex_training_data,
)

# 上层扩散模型 - MLP版本
from .trainable_obstacle_diffusion import (
    TrainableObstacleDiffusion,
    DiffusionConfig,
)

# 上层扩散模型 - Transformer版本 (性能更好)
from .transformer_obstacle_diffusion import (
    TransformerObstacleDiffusionWrapper,
    TransformerDiffusionConfig,
)

# MPD集成模块延迟导入 (避免isaacgym导入顺序问题)
def get_mpd_integration():
    """获取MPD集成模块 (延迟导入)"""
    from .mpd_integration import (
        MPDModelConfig,
        PretrainedMPDLoader,
        DynamicMPDPlanner,
        create_planner_from_config,
    )
    return {
        'MPDModelConfig': MPDModelConfig,
        'PretrainedMPDLoader': PretrainedMPDLoader,
        'DynamicMPDPlanner': DynamicMPDPlanner,
        'create_planner_from_config': create_planner_from_config,
    }

__all__ = [
    # 数据生成
    "ObstacleMotionConfig",
    "ComplexObstacleDataGenerator",
    "generate_complex_training_data",
    
    # 上层扩散 - MLP
    "TrainableObstacleDiffusion",
    "DiffusionConfig",
    
    # 上层扩散 - Transformer
    "TransformerObstacleDiffusionWrapper",
    "TransformerDiffusionConfig",
    
    # MPD集成 (延迟导入)
    "get_mpd_integration",
]

__version__ = "0.3.0"  # 双层扩散架构 + Transformer上层模型

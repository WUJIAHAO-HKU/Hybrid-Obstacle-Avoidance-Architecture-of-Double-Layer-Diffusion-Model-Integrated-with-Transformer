"""
加载已训练的障碍物扩散模型

用法:
    from scripts.load_trained_model import load_trained_diffusion_model
    model, config, training_info = load_trained_diffusion_model()
"""

import sys
sys.path.insert(0, '/home/wujiahao/mpd-build/dynamic_mpd')

import torch
from src.trainable_obstacle_diffusion import TrainableObstacleDiffusion, DiffusionConfig


def load_trained_diffusion_model(
    model_path: str = '/home/wujiahao/mpd-build/dynamic_mpd/results/trained_diffusion_model.pth',
    device: str = 'cpu'
):
    """
    加载已训练的扩散模型
    
    Args:
        model_path: 模型文件路径
        device: 设备 ('cpu' 或 'cuda')
        
    Returns:
        model: TrainableObstacleDiffusion 实例
        config: DiffusionConfig 配置
        training_info: 训练信息字典
    """
    
    print(f"Loading model from: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 恢复配置
    config = checkpoint['config']
    
    # 重建模型
    model = TrainableObstacleDiffusion(config, device)
    model.denoise_net.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 训练信息
    training_info = {
        'loss_history': checkpoint['loss_history'],
        'test_loss_history': checkpoint['test_loss_history'],
        'best_test_loss': checkpoint['best_test_loss'],
        'num_epochs': checkpoint['num_epochs'],
        'num_train_samples': checkpoint['num_train_samples'],
        'num_test_samples': checkpoint['num_test_samples']
    }
    
    model.denoise_net.eval()
    
    print(f"  [OK] Model loaded successfully!")
    print(f"  - Trained for: {training_info['num_epochs']} epochs")
    print(f"  - Training samples: {training_info['num_train_samples']}")
    print(f"  - Best test loss: {training_info['best_test_loss']:.4f}")
    print(f"  - Final train loss: {training_info['loss_history'][-1]:.4f}")
    print(f"  - Final test loss: {training_info['test_loss_history'][-1]:.4f}")
    
    return model, config, training_info


if __name__ == '__main__':
    # 测试加载
    model, config, info = load_trained_diffusion_model()
    
    print("\n=== Model Configuration ===")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Diffusion steps: {config.diffusion_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    
    # 测试预测
    print("\n=== Testing Prediction ===")
    import torch
    test_history = torch.randn(1, 8, 2)
    
    with torch.no_grad():
        prediction = model.predict(test_history, num_samples=5)
    
    print(f"  Input history: {test_history.shape}")
    print(f"  Predictions: {prediction.shape}")
    print(f"  [OK] Model working correctly!")

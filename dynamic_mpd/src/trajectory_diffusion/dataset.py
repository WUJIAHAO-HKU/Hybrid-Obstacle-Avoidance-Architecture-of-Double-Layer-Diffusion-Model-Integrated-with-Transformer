"""
ETH/UCY 行人轨迹数据集

支持ETH和UCY数据集的加载和预处理，用于训练轨迹预测模型。

数据集:
- ETH: eth, hotel
- UCY: univ, zara1, zara2

参考:
- Social Force Model for Pedestrian Dynamics (Helbing & Molnar, 1995)
- Social GAN (Gupta et al., CVPR 2018)

Author: Dynamic MPD Project
Date: 2026-01-23
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def read_file(path: str, delim: str = '\t') -> np.ndarray:
    """读取轨迹文件 (统一格式: frame, ped_id, x, y)"""
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            # 尝试不同的分隔符
            if delim in line:
                row = [float(x) for x in line.split(delim) if x.strip()]
            else:
                row = [float(x) for x in line.split() if x.strip()]
            if len(row) >= 4:
                data.append(row[:4])  # 只取 frame, ped_id, x, y
    
    return np.array(data)


def get_homography_matrix(dataset_name: str) -> Optional[np.ndarray]:
    """获取像素到世界坐标的变换矩阵"""
    # 大多数情况下数据已经是世界坐标，返回None
    # 如果需要，可以添加特定数据集的变换矩阵
    return None


class ETHUCYDataset(Dataset):
    """
    ETH/UCY 行人轨迹数据集
    
    数据格式: [frame_id, ped_id, x, y]
    
    特性:
    - 支持leave-one-out交叉验证
    - 自动处理邻居轨迹
    - 支持数据增强
    """
    
    DATASETS = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    
    def __init__(
        self,
        data_dir: str,
        datasets: List[str] = None,
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 1,
        max_neighbors: int = 10,
        delim: str = '\t',
        normalize: bool = True,
        augment: bool = False,
        phase: str = 'train',
    ):
        """
        Args:
            data_dir: 数据目录
            datasets: 要加载的数据集列表 (默认全部)
            obs_len: 观测长度
            pred_len: 预测长度
            skip: 帧跳步
            min_ped: 最小行人数
            max_neighbors: 最大邻居数
            delim: 文件分隔符
            normalize: 是否归一化 (减去起点)
            augment: 是否数据增强
            phase: 'train' or 'test'
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.min_ped = min_ped
        self.max_neighbors = max_neighbors
        self.delim = delim
        self.normalize = normalize
        self.augment = augment
        self.phase = phase
        
        if datasets is None:
            datasets = self.DATASETS
        
        # 存储数据
        self.trajectories = []  # List of [seq_len, 2]
        self.neighbors = []      # List of [num_neighbors, seq_len, 2]
        self.scene_ids = []      # 场景标识
        
        # 加载所有数据集
        for dataset in datasets:
            self._load_dataset(dataset)
            
        print(f"加载完成: {len(self.trajectories)} 条轨迹")
        
    def _load_dataset(self, dataset_name: str):
        """加载单个数据集"""
        # 尝试不同的文件名格式 (支持预处理后的统一格式)
        possible_paths = [
            os.path.join(self.data_dir, f'{dataset_name}.txt'),
            os.path.join(self.data_dir, dataset_name, f'{dataset_name}.txt'),
            os.path.join(self.data_dir, dataset_name, 'true_pos_.csv'),
            os.path.join(self.data_dir, f'{dataset_name}_train.txt'),
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
                
        if data_path is None:
            print(f"警告: 找不到数据集 {dataset_name}, 跳过")
            return
            
        print(f"加载: {data_path}")
        
        # 读取数据 (统一格式: frame, ped_id, x, y)
        try:
            data = read_file(data_path, self.delim)
        except Exception as e:
            print(f"读取错误: {e}")
            return
        
        if len(data) == 0 or data.shape[1] < 4:
            print(f"数据格式错误: {dataset_name}")
            return
            
        # 提取列 (统一格式)
        frames = data[:, 0].astype(int)
        ped_ids = data[:, 1].astype(int)
        xs = data[:, 2]
        ys = data[:, 3]
        
        unique_frames = np.unique(frames)
        frame_to_idx = {f: i for i, f in enumerate(unique_frames)}
        
        unique_peds = np.unique(ped_ids)
        
        # 构建行人轨迹字典
        ped_trajectories = {}
        for ped_id in unique_peds:
            mask = ped_ids == ped_id
            ped_frames = frames[mask]
            ped_xs = xs[mask]
            ped_ys = ys[mask]
            
            ped_trajectories[ped_id] = {
                'frames': ped_frames,
                'xs': ped_xs,
                'ys': ped_ys,
            }
        
        # 构建帧到行人的映射 (用于快速邻居搜索)
        frame_to_peds = {}
        for i, (frame, ped_id) in enumerate(zip(frames, ped_ids)):
            if frame not in frame_to_peds:
                frame_to_peds[frame] = set()
            frame_to_peds[frame].add(ped_id)
        
        # 提取有效序列
        for ped_id, ped_data in ped_trajectories.items():
            ped_frames = ped_data['frames']
            
            # 检查是否有足够的帧
            if len(ped_frames) < self.seq_len:
                continue
            
            # 计算典型帧间隔 (基于中位数)
            if len(ped_frames) > 1:
                typical_gap = int(np.median(np.diff(ped_frames)))
            else:
                typical_gap = 1
                
            # 滑动窗口提取序列
            for start_idx in range(0, len(ped_frames) - self.seq_len + 1, self.skip):
                frame_window = ped_frames[start_idx:start_idx + self.seq_len]
                
                # 检查帧是否大致连续 (允许2倍的典型间隔)
                frame_diffs = np.diff(frame_window)
                max_allowed_gap = max(typical_gap * 2, 20)  # 允许最大20帧间隔
                if not np.all(frame_diffs <= max_allowed_gap):
                    continue
                
                # 提取轨迹
                xs_window = ped_data['xs'][start_idx:start_idx + self.seq_len]
                ys_window = ped_data['ys'][start_idx:start_idx + self.seq_len]
                traj = np.stack([xs_window, ys_window], axis=1)  # [seq_len, 2]
                
                # 提取邻居轨迹 (使用优化的函数)
                neighbors = self._get_neighbors(
                    ped_id, frame_window, ped_trajectories, frame_to_peds
                )
                
                self.trajectories.append(traj)
                self.neighbors.append(neighbors)
                self.scene_ids.append(dataset_name)
                
    def _get_neighbors(
        self,
        target_ped: int,
        frames: np.ndarray,
        ped_trajectories: Dict,
        frame_to_peds: Dict,
    ) -> np.ndarray:
        """获取邻居轨迹 (优化版本)"""
        neighbors = []
        
        # 找出在这些帧内出现的所有行人
        candidate_peds = set()
        for frame in frames:
            if frame in frame_to_peds:
                candidate_peds.update(frame_to_peds[frame])
        candidate_peds.discard(target_ped)
        
        for ped_id in candidate_peds:
            if ped_id not in ped_trajectories:
                continue
            ped_data = ped_trajectories[ped_id]
            ped_frames = ped_data['frames']
            
            # 构建邻居轨迹 (可能有缺失)
            neighbor_traj = np.zeros((len(frames), 2))
            valid_mask = np.zeros(len(frames), dtype=bool)
            
            # 使用集合加速查找
            ped_frame_set = set(ped_frames)
            frame_to_idx_map = {f: i for i, f in enumerate(ped_frames)}
            
            for i, frame in enumerate(frames):
                if frame in ped_frame_set:
                    idx = frame_to_idx_map[frame]
                    neighbor_traj[i, 0] = ped_data['xs'][idx]
                    neighbor_traj[i, 1] = ped_data['ys'][idx]
                    valid_mask[i] = True
                    
            # 只保留有足够可见性的邻居 (至少一半帧可见)
            if valid_mask.sum() >= len(frames) // 2:
                # 简单插值填充缺失帧
                if not valid_mask.all():
                    valid_indices = np.where(valid_mask)[0]
                    for i in range(len(frames)):
                        if not valid_mask[i]:
                            nearest = valid_indices[np.argmin(np.abs(valid_indices - i))]
                            neighbor_traj[i] = neighbor_traj[nearest]
                            
                neighbors.append(neighbor_traj)
                
            if len(neighbors) >= self.max_neighbors:
                break
                
        if len(neighbors) == 0:
            neighbors = np.zeros((1, len(frames), 2))  # 占位
        else:
            neighbors = np.stack(neighbors, axis=0)
            
        # 填充到max_neighbors
        if len(neighbors) < self.max_neighbors:
            padding = np.zeros((self.max_neighbors - len(neighbors), len(frames), 2))
            neighbors = np.concatenate([neighbors, padding], axis=0)
        else:
            neighbors = neighbors[:self.max_neighbors]
            
        return neighbors
        
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            dict with keys:
                - history: [obs_len, 2]
                - future: [pred_len, 2]
                - neighbors_history: [max_neighbors, obs_len, 2]
                - neighbors_future: [max_neighbors, pred_len, 2]
        """
        traj = self.trajectories[idx].copy()
        neighbors = self.neighbors[idx].copy()
        
        # 分割历史和未来
        history = traj[:self.obs_len]
        future = traj[self.obs_len:]
        neighbors_history = neighbors[:, :self.obs_len, :]
        neighbors_future = neighbors[:, self.obs_len:, :]
        
        # 归一化 (减去起点)
        if self.normalize:
            origin = history[0].copy()
            history = history - origin
            future = future - origin
            neighbors_history = neighbors_history - origin
            neighbors_future = neighbors_future - origin
            
        # 数据增强
        if self.augment and self.phase == 'train':
            history, future, neighbors_history, neighbors_future = \
                self._augment(history, future, neighbors_history, neighbors_future)
        
        return {
            'history': torch.from_numpy(history).float(),
            'future': torch.from_numpy(future).float(),
            'neighbors_history': torch.from_numpy(neighbors_history).float(),
            'neighbors_future': torch.from_numpy(neighbors_future).float(),
        }
    
    def _augment(
        self,
        history: np.ndarray,
        future: np.ndarray,
        neighbors_history: np.ndarray,
        neighbors_future: np.ndarray,
    ) -> Tuple:
        """数据增强: 随机旋转和翻转"""
        # 随机旋转
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-np.pi, np.pi)
            rot_mat = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            history = history @ rot_mat.T
            future = future @ rot_mat.T
            neighbors_history = np.einsum('ijk,lk->ijl', neighbors_history, rot_mat)
            neighbors_future = np.einsum('ijk,lk->ijl', neighbors_future, rot_mat)
            
        # 随机翻转
        if np.random.rand() > 0.5:
            history[:, 0] *= -1
            future[:, 0] *= -1
            neighbors_history[:, :, 0] *= -1
            neighbors_future[:, :, 0] *= -1
            
        return history, future, neighbors_history, neighbors_future


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    obs_len: int = 8,
    pred_len: int = 12,
    train_datasets: List[str] = None,
    test_datasets: List[str] = None,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批大小
        obs_len: 观测长度
        pred_len: 预测长度
        train_datasets: 训练数据集列表
        test_datasets: 测试数据集列表
        num_workers: 数据加载线程数
        
    Returns:
        train_loader, test_loader
    """
    if train_datasets is None:
        train_datasets = ['eth', 'hotel', 'univ', 'zara1']
    if test_datasets is None:
        test_datasets = ['zara2']
        
    train_dataset = ETHUCYDataset(
        data_dir=data_dir,
        datasets=train_datasets,
        obs_len=obs_len,
        pred_len=pred_len,
        augment=True,
        phase='train',
    )
    
    test_dataset = ETHUCYDataset(
        data_dir=data_dir,
        datasets=test_datasets,
        obs_len=obs_len,
        pred_len=pred_len,
        augment=False,
        phase='test',
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader


class SyntheticTrajectoryDataset(Dataset):
    """
    合成轨迹数据集
    
    用于测试和调试，生成简单的轨迹模式。
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        obs_len: int = 8,
        pred_len: int = 12,
        traj_dim: int = 2,
        max_neighbors: int = 10,
    ):
        self.num_samples = num_samples
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.traj_dim = traj_dim
        self.max_neighbors = max_neighbors
        
        # 预生成数据
        self.trajectories = self._generate_trajectories()
        self.neighbors = self._generate_neighbors()
        
    def _generate_trajectories(self) -> np.ndarray:
        """生成合成轨迹"""
        trajectories = []
        
        for _ in range(self.num_samples):
            # 随机选择轨迹类型
            traj_type = np.random.choice(['linear', 'curved', 'zigzag'])
            
            if traj_type == 'linear':
                # 线性轨迹
                velocity = np.random.randn(self.traj_dim) * 0.5
                traj = np.cumsum(
                    np.tile(velocity, (self.seq_len, 1)) + 
                    np.random.randn(self.seq_len, self.traj_dim) * 0.05,
                    axis=0
                )
            elif traj_type == 'curved':
                # 曲线轨迹
                t = np.linspace(0, 2 * np.pi * np.random.uniform(0.5, 2), self.seq_len)
                radius = np.random.uniform(1, 3)
                traj = np.stack([
                    radius * np.cos(t),
                    radius * np.sin(t)
                ], axis=1)
                traj += np.random.randn(self.seq_len, self.traj_dim) * 0.05
            else:
                # 折线轨迹
                traj = np.zeros((self.seq_len, self.traj_dim))
                velocity = np.random.randn(self.traj_dim) * 0.5
                for i in range(1, self.seq_len):
                    if np.random.rand() > 0.9:
                        velocity = np.random.randn(self.traj_dim) * 0.5
                    traj[i] = traj[i-1] + velocity + np.random.randn(self.traj_dim) * 0.02
                    
            trajectories.append(traj)
            
        return np.stack(trajectories, axis=0)
    
    def _generate_neighbors(self) -> np.ndarray:
        """生成邻居轨迹"""
        neighbors = []
        
        for i in range(self.num_samples):
            num_neighbors = np.random.randint(0, self.max_neighbors + 1)
            
            if num_neighbors == 0:
                neighbor = np.zeros((self.max_neighbors, self.seq_len, self.traj_dim))
            else:
                neighbor_trajs = []
                for _ in range(num_neighbors):
                    # 基于主轨迹生成附近的邻居轨迹
                    offset = np.random.randn(self.traj_dim) * 2
                    neighbor_traj = self.trajectories[i] + offset
                    neighbor_traj += np.random.randn(self.seq_len, self.traj_dim) * 0.1
                    neighbor_trajs.append(neighbor_traj)
                    
                # 填充到max_neighbors
                while len(neighbor_trajs) < self.max_neighbors:
                    neighbor_trajs.append(np.zeros((self.seq_len, self.traj_dim)))
                    
                neighbor = np.stack(neighbor_trajs, axis=0)
                
            neighbors.append(neighbor)
            
        return np.stack(neighbors, axis=0)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx].copy()
        neighbors = self.neighbors[idx].copy()
        
        # 归一化
        origin = traj[0].copy()
        traj = traj - origin
        neighbors = neighbors - origin
        
        history = traj[:self.obs_len]
        future = traj[self.obs_len:]
        neighbors_history = neighbors[:, :self.obs_len, :]
        neighbors_future = neighbors[:, self.obs_len:, :]
        
        return {
            'history': torch.from_numpy(history).float(),
            'future': torch.from_numpy(future).float(),
            'neighbors_history': torch.from_numpy(neighbors_history).float(),
            'neighbors_future': torch.from_numpy(neighbors_future).float(),
        }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("数据集测试")
    print("=" * 60)
    
    # 测试合成数据集
    print("\n1. 合成数据集测试")
    dataset = SyntheticTrajectoryDataset(
        num_samples=100,
        obs_len=8,
        pred_len=12,
    )
    print(f"  数据集大小: {len(dataset)}")
    
    sample = dataset[0]
    print(f"  history: {sample['history'].shape}")
    print(f"  future: {sample['future'].shape}")
    print(f"  neighbors_history: {sample['neighbors_history'].shape}")
    print(f"  neighbors_future: {sample['neighbors_future'].shape}")
    
    # DataLoader测试
    print("\n2. DataLoader测试")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    print(f"  batch history: {batch['history'].shape}")
    print(f"  batch future: {batch['future'].shape}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

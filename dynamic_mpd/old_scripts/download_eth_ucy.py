#!/usr/bin/env python3
"""
下载ETH/UCY行人轨迹数据集

数据集来源: Social GAN (Stanford)
数据格式: frame_id, ped_id, x, y

Author: Dynamic MPD Project
Date: 2026-01-23
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """下载文件"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    # 数据目录
    data_dir = Path(__file__).parent.parent / 'data' / 'eth_ucy'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("下载ETH/UCY数据集")
    print("=" * 60)
    print(f"保存目录: {data_dir}")
    
    # 尝试多个数据源
    # 方案1: Trajectron++处理过的数据 (推荐)
    # 方案2: 原始UCY数据集
    
    # 使用原始数据集格式 (更简单)
    print("\n尝试从多个源下载...")
    
    # 数据集URL列表 (多个备选源)
    datasets = {
        'eth': [
            'https://github.com/StanfordASL/Trajectron-plus-plus/raw/master/experiments/pedestrians/raw/eth/train/biwi_eth.txt',
            'https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/master/data/train/biwi_eth.txt',
        ],
        'hotel': [
            'https://github.com/StanfordASL/Trajectron-plus-plus/raw/master/experiments/pedestrians/raw/eth/train/biwi_hotel.txt',
            'https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/master/data/train/biwi_hotel.txt',
        ],
        'univ': [
            'https://github.com/StanfordASL/Trajectron-plus-plus/raw/master/experiments/pedestrians/raw/ucy/students03/students003.txt',
            'https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/master/data/train/students003.txt',
        ],
        'zara1': [
            'https://github.com/StanfordASL/Trajectron-plus-plus/raw/master/experiments/pedestrians/raw/ucy/zara01/crowds_zara01.txt',
            'https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/master/data/train/crowds_zara01.txt',
        ],
        'zara2': [
            'https://github.com/StanfordASL/Trajectron-plus-plus/raw/master/experiments/pedestrians/raw/ucy/zara02/crowds_zara02.txt',
            'https://raw.githubusercontent.com/vita-epfl/trajnetplusplusdata/master/data/train/crowds_zara02.txt',
        ],
    }
    
    # 下载每个数据集
    for name, urls in datasets.items():
        output_path = data_dir / f'{name}.txt'
        
        if output_path.exists():
            print(f"\n✓ {name}.txt 已存在，跳过")
            continue
            
        print(f"\n下载 {name}...")
        success = False
        
        # 尝试所有URL
        for i, url in enumerate(urls):
            try:
                print(f"  尝试源 {i+1}/{len(urls)}: {url.split('/')[-1]}")
                download_url(url, str(output_path))
                
                # 验证文件
                with open(output_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        print(f"  ✓ 成功! {len(lines)} 行数据")
                        success = True
                        break
                    else:
                        print(f"  ✗ 文件为空")
                        output_path.unlink()
                        
            except Exception as e:
                print(f"  ✗ 失败: {str(e)[:50]}")
                if output_path.exists():
                    output_path.unlink()
        
        if not success:
            print(f"  ⚠ {name} 所有源均失败")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("下载完成! 数据集统计:")
    print("=" * 60)
    
    total_lines = 0
    for name in datasets.keys():
        file_path = data_dir / f'{name}.txt'
        if file_path.exists():
            with open(file_path, 'r') as f:
                num_lines = len(f.readlines())
                total_lines += num_lines
                
                # 读取数据统计
                f.seek(0)
                frames = set()
                peds = set()
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        frames.add(int(float(parts[0])))
                        peds.add(int(float(parts[1])))
                
                print(f"{name:10s}: {num_lines:6d} 行, {len(frames):4d} 帧, {len(peds):3d} 行人")
    
    print(f"{'总计':10s}: {total_lines:6d} 行")
    print("\n数据格式: frame_id  ped_id  x  y")
    print(f"数据位置: {data_dir}")
    
    # 创建数据集分割说明
    split_info = data_dir / 'README.txt'
    with open(split_info, 'w') as f:
        f.write("ETH/UCY Pedestrian Trajectory Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write("数据来源: Social GAN (Stanford)\n\n")
        f.write("数据集:\n")
        f.write("- eth.txt: ETH dataset\n")
        f.write("- hotel.txt: Hotel dataset\n")
        f.write("- univ.txt: University students\n")
        f.write("- zara1.txt: Zara1 dataset\n")
        f.write("- zara2.txt: Zara2 dataset\n\n")
        f.write("数据格式:\n")
        f.write("frame_id  ped_id  x  y\n\n")
        f.write("常用分割 (Leave-one-out):\n")
        f.write("- Train: 4个数据集, Test: 1个数据集\n")
        f.write("- 例如: Train=[eth,hotel,univ,zara1], Test=[zara2]\n")
    
    print(f"\n说明文件已保存: {split_info}")


if __name__ == '__main__':
    main()

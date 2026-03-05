#!/usr/bin/env python3
"""
ROSOrin + 双层扩散模型 Isaac Sim导航演示

完全基于 rosorin_lidar_navigation.py 的算法架构，替换仿真层：
- 智能体：ROSOrin 麦克纳姆轮小车 (从 URDF 转换)
- 障碍物：动态移动圆柱体
- 仿真：Isaac Sim 物理引擎
- 算法：双层扩散模型 (Transformer上层 + MPD下层)

运行方式:
    cd "/home/wujiahao/ROSORIN_CAR and Reasearch/Diffusion-Guided Path Optimization (DGPO)"
    ./isaaclab_runner.sh "/path/to/dynamic_mpd/scripts/rosorin_demo/isaac_sim_navigation.py"

作者: Dynamic MPD Project
日期: 2026-01-30
"""

# ============================================================
# 1. AppLauncher 必须最先初始化 (Isaac Lab 要求)
# ============================================================
import argparse
parser = argparse.ArgumentParser(description="ROSOrin Diffusion Navigation Demo in Isaac Sim")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--max_steps", type=int, default=800, help="Maximum simulation steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--playback_speed", type=float, default=1.0, help="Playback speed multiplier (higher = faster)")
parser.add_argument("--skip_compute", action="store_true", help="Skip trajectory computation phase visualization")
parser.add_argument("--capture", action="store_true", help="Enable trajectory capture for composite image")
parser.add_argument("--capture_interval", type=int, default=15, help="Capture frame every N steps")
args = parser.parse_args()

import os
import sys

# 添加Isaac Lab source到路径
isaac_lab_path = os.environ.get("ISAAC_LAB_PATH", "/home/wujiahao/IsaacLab")
if os.path.exists(os.path.join(isaac_lab_path, "source")):
    sys.path.insert(0, os.path.join(isaac_lab_path, "source"))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ============================================================
# 2. 现在可以安全导入其他模块
# ============================================================
import numpy as np
import torch
import math
import time
from collections import deque
from typing import List, Dict, Tuple, Optional

# Isaac imports
import omni.usd
import omni.kit.app
from pxr import UsdGeom, Gf, UsdPhysics, Sdf, UsdLux, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg

# Matplotlib for saving visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import io

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
dynamic_mpd_root = os.path.abspath(os.path.join(script_dir, '..', '..'))


# ============================================================
# 轨迹跟踪截图器
# ============================================================
class TrajectoryCapture:
    """固定视角截图并合成轨迹跟踪图"""
    
    def __init__(self, output_dir: str, capture_interval: int = 10):
        """
        Args:
            output_dir: 输出目录
            capture_interval: 每隔多少步截一次图
        """
        self.output_dir = output_dir
        self.capture_interval = capture_interval
        self.frames = []  # 存储截图
        self.robot_positions = []  # 存储对应的机器人位置
        self.step_indices = []  # 存储对应的步数
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
        
        # Viewport 相关
        self._viewport = None
        self._viewport_api = None
        
    def setup_camera(self, stage, arena_size: float):
        """设置固定俯视相机"""
        import omni.kit.viewport.utility as viewport_utils
        
        # 获取 viewport
        self._viewport = viewport_utils.get_active_viewport()
        
        if self._viewport is None:
            print("  [WARN] No active viewport found, skipping camera setup")
            return False
        
        # 创建固定相机
        camera_path = "/World/TrackingCamera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        
        # 设置相机参数 - 斜上方45度俯视
        height = arena_size * 1.5
        distance = arena_size * 1.2
        
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        
        # 位置：斜上方
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(-distance * 0.7, -distance * 0.7, height))
        
        # 旋转：朝向场地中心
        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(55, 0, -45))  # 俯视角度
        
        # 相机属性
        camera.GetFocalLengthAttr().Set(24.0)
        camera.GetHorizontalApertureAttr().Set(36.0)
        
        # 切换到这个相机
        try:
            self._viewport.set_active_camera(camera_path)
            print(f"  [OK] Tracking camera created at {camera_path}")
            return True
        except Exception as e:
            print(f"  [WARN] Failed to set camera: {e}")
            return False
    
    def capture_frame(self, step: int, robot_pos: np.ndarray):
        """截取当前帧 - 使用 viewport API"""
        if step % self.capture_interval != 0:
            return
        
        frame_path = os.path.join(self.output_dir, 'frames', f'frame_{step:04d}.png')
        
        try:
            import omni.kit.viewport.utility as viewport_utils
            
            viewport = viewport_utils.get_active_viewport()
            if viewport is None:
                print(f"    [WARN] No viewport at step {step}")
                return
            
            # 使用 viewport 的 capture_frame_to_file 方法
            viewport.capture_frame(frame_path)
            
            # 等待文件写入
            for _ in range(5):
                omni.kit.app.get_app().update()
            
            if os.path.exists(frame_path):
                self.frames.append(frame_path)
                self.robot_positions.append(robot_pos.copy())
                self.step_indices.append(step)
                print(f"    [Captured] frame_{step:04d}.png")
                
        except Exception as e:
            print(f"    [WARN] Capture failed at step {step}: {e}")
    
    def capture_frame_simple(self, step: int, robot_pos: np.ndarray):
        """截图 - 尝试多种方法"""
        if step % self.capture_interval != 0:
            return
        
        frame_path = os.path.join(self.output_dir, 'frames', f'frame_{step:04d}.png')
        
        # 首先检查该帧是否已存在
        if os.path.exists(frame_path):
            print(f"    [Skip] frame_{step:04d}.png already exists")
            return
        
        print(f"    [Capture] Trying to capture step {step}...")
        
        # 方法1：使用 legacy viewport
        try:
            from omni.kit.viewport_legacy import acquire_viewport_interface
            
            vp_iface = acquire_viewport_interface()
            vp = vp_iface.get_viewport_window(None)
            
            if vp is not None:
                vp.capture_frame(frame_path)
                
                for _ in range(15):
                    omni.kit.app.get_app().update()
                
                import time as t
                t.sleep(0.1)
                
                if os.path.exists(frame_path):
                    self.frames.append(frame_path)
                    self.robot_positions.append(robot_pos.copy())
                    self.step_indices.append(step)
                    print(f"      [OK] Method1: legacy viewport")
                    return
                else:
                    print(f"      [FAIL] Method1: file not created")
            else:
                print(f"      [FAIL] Method1: no viewport window")
            
        except ImportError as e:
            print(f"      [SKIP] Method1: {e}")
        except Exception as e:
            print(f"      [FAIL] Method1: {e}")
        
        # 方法2：使用 Kit Capture 扩展
        try:
            import omni.kit.capture.viewport
            
            capture_ext = omni.kit.capture.viewport.CaptureExtension.get_instance()
            if capture_ext:
                capture_ext.capture_frame(frame_path)
                
                for _ in range(15):
                    omni.kit.app.get_app().update()
                
                import time as t
                t.sleep(0.1)
                
                if os.path.exists(frame_path):
                    self.frames.append(frame_path)
                    self.robot_positions.append(robot_pos.copy())
                    self.step_indices.append(step)
                    print(f"      [OK] Method2: Kit Capture")
                    return
                else:
                    print(f"      [FAIL] Method2: file not created")
            else:
                print(f"      [FAIL] Method2: no capture extension")
                    
        except ImportError as e:
            print(f"      [SKIP] Method2: {e}")
        except Exception as e:
            print(f"      [FAIL] Method2: {e}")
        
        # 方法3：直接使用 Omniverse 截图 API
        try:
            import omni.renderer_capture
            
            renderer_capture = omni.renderer_capture.acquire_renderer_capture_interface()
            renderer_capture.capture_next_frame_swapchain(frame_path)
            
            for _ in range(15):
                omni.kit.app.get_app().update()
            
            import time as t
            t.sleep(0.1)
            
            if os.path.exists(frame_path):
                self.frames.append(frame_path)
                self.robot_positions.append(robot_pos.copy())
                self.step_indices.append(step)
                print(f"      [OK] Method3: renderer_capture")
                return
            else:
                print(f"      [FAIL] Method3: file not created")
                
        except ImportError as e:
            print(f"      [SKIP] Method3: {e}")
        except Exception as e:
            print(f"      [FAIL] Method3: {e}")
        
        # 方法4：尝试 Replicator 方法
        captured = self.capture_with_replicator(step, robot_pos)
        if captured:
            return
        
        # 方法5：尝试 numpy/sensor 方法
        captured = self.capture_with_numpy(step, robot_pos)
        if captured:
            return
        
        # 所有方法都失败了
        print(f"      [ERROR] All capture methods failed for step {step}")
    
    def _capture_with_renderer(self, step: int, robot_pos: np.ndarray, frame_path: str):
        """备用方法：使用 omni.kit.window.viewport 截图"""
        try:
            from omni.kit.window.viewport import get_viewport_interface
            
            vp_iface = get_viewport_interface()
            vp_window = vp_iface.get_viewport_window()
            
            if vp_window:
                vp_window.capture_frame(frame_path)
                
                for _ in range(5):
                    omni.kit.app.get_app().update()
                
                if os.path.exists(frame_path):
                    self.frames.append(frame_path)
                    self.robot_positions.append(robot_pos.copy())
                    self.step_indices.append(step)
                    print(f"    [Captured via renderer] frame_{step:04d}.png")
        except Exception as e2:
            pass
    
    def capture_with_replicator(self, step: int, robot_pos: np.ndarray) -> bool:
        """使用 Replicator API 截图 - 最可靠的方法"""
        if step % self.capture_interval != 0:
            return False
        
        frame_path = os.path.join(self.output_dir, 'frames', f'frame_{step:04d}.png')
        
        try:
            import omni.replicator.core as rep
            
            # 创建或获取渲染产品
            if not hasattr(self, '_render_product'):
                self._render_product = rep.create.render_product(
                    self.camera_path,
                    resolution=(self.resolution[0], self.resolution[1])
                )
                self._writer = rep.WriterRegistry.get("BasicWriter")
                self._writer.initialize(
                    output_dir=os.path.join(self.output_dir, 'frames'),
                    rgb=True
                )
                self._writer.attach([self._render_product])
            
            # 触发渲染
            rep.orchestrator.step()
            
            for _ in range(10):
                omni.kit.app.get_app().update()
            
            import time as t
            t.sleep(0.1)
            
            # 检查输出
            frames_dir = os.path.join(self.output_dir, 'frames')
            latest_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
            
            if latest_files:
                latest = os.path.join(frames_dir, latest_files[-1])
                if latest not in self.frames:
                    # 重命名为我们的格式
                    new_name = frame_path
                    os.rename(latest, new_name)
                    self.frames.append(new_name)
                    self.robot_positions.append(robot_pos.copy())
                    self.step_indices.append(step)
                    print(f"      [OK] Method4: Replicator")
                    return True
            
            print(f"      [FAIL] Method4: no output files")
            return False
                    
        except ImportError as e:
            print(f"      [SKIP] Method4: {e}")
            return False
        except Exception as e:
            print(f"      [FAIL] Method4: {e}")
            return False

    def capture_with_numpy(self, step: int, robot_pos: np.ndarray) -> bool:
        """使用 numpy 数组获取渲染结果并保存"""
        if step % self.capture_interval != 0:
            return False
        
        frame_path = os.path.join(self.output_dir, 'frames', f'frame_{step:04d}.png')
        
        try:
            import omni.kit.viewport.utility as viewport_utils
            import numpy as np
            
            viewport = viewport_utils.get_active_viewport()
            if viewport is None:
                print(f"      [FAIL] Method5: no viewport")
                return False
            
            # 尝试获取渲染纹理
            try:
                from omni.isaac.sensor import Camera
                
                # 如果没有相机传感器，创建一个
                if not hasattr(self, '_sensor_camera'):
                    self._sensor_camera = Camera(
                        prim_path=self.camera_path,
                        resolution=(self.resolution[0], self.resolution[1])
                    )
                    self._sensor_camera.initialize()
                
                # 获取 RGBA 数据
                rgba = self._sensor_camera.get_rgba()
                
                if rgba is not None and len(rgba) > 0:
                    img = Image.fromarray(rgba)
                    img.save(frame_path)
                    
                    self.frames.append(frame_path)
                    self.robot_positions.append(robot_pos.copy())
                    self.step_indices.append(step)
                    print(f"      [OK] Method5: sensor camera")
                    return True
                else:
                    print(f"      [FAIL] Method5: no rgba data")
                    
            except ImportError as e:
                print(f"      [SKIP] Method5: {e}")
            
        except Exception as e:
            print(f"      [FAIL] Method5: {e}")
        
        return False
    
    def create_trajectory_composite(self, goal: np.ndarray) -> str:
        """合成轨迹跟踪图"""
        output_path = os.path.join(self.output_dir, 'trajectory_composite.png')
        
        if not self.frames:
            print("  [WARN] No frames captured for composite")
            return None
        
        # 加载所有帧
        images = []
        for f in self.frames:
            if isinstance(f, str):
                if os.path.exists(f):
                    images.append(Image.open(f).convert('RGBA'))
            else:
                images.append(f.convert('RGBA'))
        
        if not images:
            print("  [WARN] No valid images for composite")
            return None
        
        # 创建合成图像
        base_img = images[0].copy()
        width, height = base_img.size
        
        # 方法1: 最大值混合 - 突出显示所有时刻的小车位置
        composite = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        for i, img in enumerate(images):
            # 计算透明度 - 越新的帧越不透明
            alpha = int(80 + 175 * (i / len(images)))
            
            # 调整透明度
            r, g, b, a = img.split()
            a = a.point(lambda x: min(x, alpha))
            img_alpha = Image.merge('RGBA', (r, g, b, a))
            
            # 叠加
            composite = Image.alpha_composite(composite, img_alpha)
        
        # 最后一帧完全不透明
        composite = Image.alpha_composite(composite, images[-1])
        
        # 转为 RGB 保存
        final = Image.new('RGB', (width, height), (40, 40, 50))
        final.paste(composite, mask=composite.split()[3])
        final.save(output_path, quality=95)
        
        print(f"  [OK] Trajectory composite saved: {output_path}")
        print(f"       Total frames: {len(images)}")
        
        return output_path
    
    def create_trajectory_montage(self, cols: int = 5) -> str:
        """创建轨迹蒙太奇 - 多帧拼接"""
        output_path = os.path.join(self.output_dir, 'trajectory_montage.png')
        
        # 加载所有帧
        images = []
        for f in self.frames:
            if isinstance(f, str):
                if os.path.exists(f):
                    images.append(Image.open(f))
            else:
                images.append(f)
        
        if not images:
            return None
        
        # 选择关键帧（均匀采样）
        n_frames = min(15, len(images))
        indices = np.linspace(0, len(images) - 1, n_frames, dtype=int)
        key_frames = [images[i] for i in indices]
        
        # 缩放
        thumb_width = 320
        thumb_height = int(thumb_width * key_frames[0].height / key_frames[0].width)
        
        thumbnails = [img.resize((thumb_width, thumb_height), Image.LANCZOS) 
                     for img in key_frames]
        
        # 拼接
        rows = (len(thumbnails) + cols - 1) // cols
        montage_width = cols * thumb_width
        montage_height = rows * thumb_height
        
        montage = Image.new('RGB', (montage_width, montage_height), (30, 30, 40))
        
        for i, thumb in enumerate(thumbnails):
            x = (i % cols) * thumb_width
            y = (i // cols) * thumb_height
            montage.paste(thumb, (x, y))
        
        montage.save(output_path, quality=90)
        print(f"  [OK] Trajectory montage saved: {output_path}")
        
        return output_path
sys.path.insert(0, dynamic_mpd_root)

# 随机种子 (从命令行参数获取)
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ROSOrin USD 模型路径 (从 DGPO 项目获取)
DGPO_ROOT = "/home/wujiahao/ROSORIN_CAR and Reasearch/Diffusion-Guided Path Optimization (DGPO)"
ROSORIN_USD_PATH = os.path.join(DGPO_ROOT, "data/assets/rosorin/rosorin.usd")


# ============================================================
# 3. ROSOrin 小车参数 (来自官方规格)
# ============================================================
class ROSOrinConfig:
    """ROSOrin 麦克纳姆轮小车配置"""
    # 底盘尺寸
    length = 0.23  # 车身长度 (m)
    width = 0.20   # 车身宽度 (m)
    height = 0.12  # 车身高度 (m)
    wheel_radius = 0.033  # 轮子半径 (m)
    wheelbase = 0.206     # 轴距 (m)
    
    # 运动限制
    max_linear_velocity = 0.5   # 最大线速度 (m/s)
    max_angular_velocity = 2.0  # 最大角速度 (rad/s)
    
    # LiDAR 参数 (MS200)
    lidar_range = 12.0      # 最大探测距离 (m)
    lidar_num_rays = 360    # 射线数量
    lidar_height = 0.136    # LiDAR 安装高度 (m)
    
    # 仿真缩放 (场地设置)
    arena_size = 4.0  # 场地大小 (m)
    sim_lidar_range = 3.5  # 仿真中的感知距离
    
    # USD 模型路径
    usd_path = ROSORIN_USD_PATH


# ============================================================
# 4. 双层扩散模型导航器 (直接复用 rosorin_lidar_navigation 的逻辑)
# ============================================================
class DualDiffusionNavigator:
    """双层扩散模型导航器 - 与 rosorin_lidar_navigation.py 保持一致"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tensor_args = {'device': device, 'dtype': torch.float32}
        
        # 模型
        self.upper_model = None
        self.upper_config = None
        self.use_transformer = False
        
        # 规划参数 (与2D版本一致)
        self.replan_interval = 5
        self.cached_trajectory = None
        self.trajectory_progress = 0
        self.last_replan_step = -100
        self.known_obstacle_ids = set()
        
        # 障碍物历史
        self.obstacle_history = {}
        self.history_length = 8
        
    def load_models(self):
        """加载双层扩散模型"""
        print("\n  Loading dual diffusion models...")
        
        # 尝试加载 Transformer 上层模型
        try:
            from src.transformer_obstacle_diffusion import TransformerObstacleDiffusionWrapper
            
            upper_path = os.path.join(dynamic_mpd_root, 'trained_models', 
                                     'best_transformer_obstacle_model.pt')
            if os.path.exists(upper_path):
                self.upper_model = TransformerObstacleDiffusionWrapper.load_from_checkpoint(
                    upper_path, device=self.device
                )
                self.use_transformer = True
                print(f"    [Upper] Transformer model loaded!")
            else:
                print(f"    [Upper] Model not found: {upper_path}")
                
        except Exception as e:
            print(f"    [Upper] Failed to load: {e}")
        
        # 下层模型：这里使用势场法作为简化替代
        # 完整版本可以加载 MPD 模型
        print(f"    [Lower] Using potential field planning")
        
    def update_obstacle_history(self, obs_id: int, position: np.ndarray):
        """更新单个障碍物的历史"""
        if obs_id not in self.obstacle_history:
            self.obstacle_history[obs_id] = deque(maxlen=self.history_length)
        self.obstacle_history[obs_id].append(position.copy())
        
    def predict_obstacle_motion(self, obs_id: int) -> Optional[np.ndarray]:
        """使用上层Transformer模型预测障碍物运动"""
        if obs_id not in self.obstacle_history:
            return None
            
        history = self.obstacle_history[obs_id]
        if len(history) < 4:
            return None
            
        if self.use_transformer and self.upper_model is not None:
            try:
                hist_array = np.array(list(history))[-8:]
                if len(hist_array) < 8:
                    padding = np.tile(hist_array[0], (8 - len(hist_array), 1))
                    hist_array = np.vstack([padding, hist_array])
                
                hist_tensor = torch.tensor(hist_array, device=self.device, dtype=torch.float32)
                
                with torch.no_grad():
                    mean_pred, _, _ = self.upper_model.predict(hist_tensor, num_samples=5)
                
                return mean_pred.cpu().numpy()
            except Exception as e:
                pass
        
        # 回退：线性外推
        vel = history[-1] - history[-2]
        return np.array([history[-1] + vel * (i+1) * 0.1 for i in range(12)])
    
    def plan_trajectory(self, robot_pos: np.ndarray, goal: np.ndarray,
                       obstacles: List[Dict], num_samples: int = 16) -> np.ndarray:
        """规划避障轨迹 - 势场法"""
        trajectory = [robot_pos.copy()]
        current = robot_pos.copy()
        
        # 收集所有障碍物位置（当前 + 预测）
        all_obs = []
        for obs in obstacles:
            pos = np.array(obs['position'][:2])
            all_obs.append({'pos': pos, 'radius': obs.get('radius', 0.3)})
            
            # 添加预测位置
            obs_id = obs.get('id', 0)
            if obs.get('is_dynamic', False):
                pred = self.predict_obstacle_motion(obs_id)
                if pred is not None:
                    for t in [3, 6, 9]:
                        if t < len(pred):
                            all_obs.append({'pos': pred[t], 'radius': obs.get('radius', 0.3)})
        
        # 势场规划
        step_size = 0.15
        for _ in range(num_samples - 1):
            # 吸引力（朝向目标）
            to_goal = goal - current
            dist_goal = np.linalg.norm(to_goal)
            if dist_goal < 0.1:
                trajectory.append(goal.copy())
                break
            attract = to_goal / dist_goal * min(step_size, dist_goal)
            
            # 排斥力（远离障碍物）
            repel = np.zeros(2)
            for obs_data in all_obs:
                obs_pos = obs_data['pos']
                obs_r = obs_data['radius']
                to_robot = current - obs_pos
                dist = np.linalg.norm(to_robot)
                safety = obs_r + 0.3
                
                if dist < safety and dist > 0.01:
                    strength = (safety - dist) / safety
                    repel += (to_robot / dist) * strength * 0.2
            
            # 合成速度
            velocity = attract + repel
            vel_mag = np.linalg.norm(velocity)
            if vel_mag > step_size:
                velocity = velocity / vel_mag * step_size
            
            current = current + velocity
            trajectory.append(current.copy())
        
        return np.array(trajectory)
    
    def should_replan(self, current_step: int, visible_obstacles: List[Dict]) -> Tuple[bool, str]:
        """判断是否需要重规划"""
        if current_step - self.last_replan_step >= self.replan_interval:
            return True, "periodic"
        
        if self.cached_trajectory is None or self.trajectory_progress >= len(self.cached_trajectory) - 2:
            return True, "trajectory_exhausted"
        
        current_ids = set(obs.get('id', 0) for obs in visible_obstacles)
        new_obstacles = current_ids - self.known_obstacle_ids
        if new_obstacles:
            self.known_obstacle_ids = current_ids
            return True, f"new_obstacles"
        
        return False, None
    
    def get_control(self, robot_pos: np.ndarray, goal: np.ndarray,
                   visible_obstacles: List[Dict], current_step: int) -> Tuple[np.ndarray, bool, str]:
        """获取控制命令"""
        robot_pos = np.array(robot_pos[:2])
        goal = np.array(goal[:2])
        
        dist_to_goal = np.linalg.norm(robot_pos - goal)
        
        # 检查是否需要重规划
        need_replan, reason = self.should_replan(current_step, visible_obstacles)
        
        if need_replan:
            # 更新障碍物历史
            for obs in visible_obstacles:
                self.update_obstacle_history(obs.get('id', 0), np.array(obs['position'][:2]))
            
            # 规划新轨迹
            self.cached_trajectory = self.plan_trajectory(robot_pos, goal, visible_obstacles)
            self.trajectory_progress = 0
            self.last_replan_step = current_step
        
        # 紧急避障
        for obs in visible_obstacles:
            obs_pos = np.array(obs['position'][:2])
            dist = np.linalg.norm(robot_pos - obs_pos)
            safe_dist = obs.get('radius', 0.3) + 0.15
            
            if dist < safe_dist:
                avoid_dir = robot_pos - obs_pos
                if np.linalg.norm(avoid_dir) > 0.01:
                    avoid_dir = avoid_dir / np.linalg.norm(avoid_dir)
                    return avoid_dir * 0.15, True, "emergency"
        
        # 跟随轨迹
        if self.cached_trajectory is not None and self.trajectory_progress < len(self.cached_trajectory) - 1:
            lookahead = min(self.trajectory_progress + 2, len(self.cached_trajectory) - 1)
            target = self.cached_trajectory[lookahead]
            
            direction = target - robot_pos
            dist = np.linalg.norm(direction)
            
            if dist > 0.01:
                direction = direction / dist
            
            # 接近目标时混合方向
            if dist_to_goal < 1.0:
                goal_dir = goal - robot_pos
                goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
                weight = max(0.4, 1.0 - dist_to_goal / 1.0)
                direction = (1 - weight) * direction + weight * goal_dir
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            # 速度设置 - 适中速度保证流畅可见
            speed = 0.08 if dist_to_goal > 0.5 else 0.04
            self.trajectory_progress += 1
            
            return direction * speed, need_replan, reason
        else:
            direction = goal - robot_pos
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                direction = direction / dist * 0.06
            return direction, need_replan, reason


# ============================================================
# 5. Isaac Sim 场景创建
# ============================================================
class IsaacSimScene:
    """Isaac Sim 场景 - 平面 + 动态圆柱体障碍物"""
    
    def __init__(self, stage, config: ROSOrinConfig):
        self.stage = stage
        self.config = config
        
        # 场景参数
        self.arena_size = config.arena_size
        self.robot_pos = np.array([-self.arena_size/2 + 0.5, -self.arena_size/2 + 0.5, 0.0])
        self.goal_pos = np.array([self.arena_size/2 - 0.5, self.arena_size/2 - 0.5, 0.0])
        
        # 动态障碍物
        self.obstacles = []
        self.num_obstacles = 10
        
    def create_scene(self):
        """创建完整场景"""
        print("\n============================================================")
        print("  Creating Isaac Sim Navigation Scene")
        print("============================================================")
        
        self._create_ground()
        self._create_lighting()
        self._create_start_goal_markers()
        self._create_dynamic_obstacles()
        self._create_robot_visual()
        
        print("============================================================\n")
    
    def _create_ground(self):
        """创建木质地板"""
        ground_path = "/World/Ground"
        ground_prim = UsdGeom.Mesh.Define(self.stage, ground_path)
        
        # 创建带有 UV 坐标的地板网格 (用于纹理平铺)
        size = self.arena_size * 1.5
        tile_size = 0.5  # 每块木板大小
        tiles_per_side = int(size * 2 / tile_size)
        
        points = []
        face_vertex_counts = []
        face_vertex_indices = []
        uvs = []
        
        # 创建网格化地板
        idx = 0
        for i in range(tiles_per_side):
            for j in range(tiles_per_side):
                x0 = -size + i * tile_size
                x1 = x0 + tile_size
                y0 = -size + j * tile_size
                y1 = y0 + tile_size
                
                # 四个顶点
                points.extend([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)])
                face_vertex_counts.append(4)
                face_vertex_indices.extend([idx, idx+1, idx+2, idx+3])
                
                # UV 坐标 (交替翻转模拟木板纹理)
                if (i + j) % 2 == 0:
                    uvs.extend([(0, 0), (1, 0), (1, 1), (0, 1)])
                else:
                    uvs.extend([(1, 0), (0, 0), (0, 1), (1, 1)])
                
                idx += 4
        
        ground_prim.GetPointsAttr().Set(points)
        ground_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        ground_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
        
        # 设置 UV 坐标 (使用 PrimvarsAPI)
        primvars_api = UsdGeom.PrimvarsAPI(ground_prim.GetPrim())
        texcoords = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, 
                                               UsdGeom.Tokens.faceVarying)
        texcoords.Set(uvs)
        
        # 创建木质材质
        self._create_wood_material(ground_prim)
        
        print("  [OK] Wood floor created")
    
    def _create_wood_material(self, ground_prim):
        """创建木质材质 (程序化木纹)"""
        # 创建材质
        material_path = "/World/Materials/WoodFloor"
        material = UsdShade.Material.Define(self.stage, material_path)
        
        # 创建 PBR Shader
        shader_path = material_path + "/Shader"
        shader = UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        # 木质颜色 (温暖的橡木色)
        wood_color = Gf.Vec3f(0.55, 0.35, 0.20)  # 橡木色
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(wood_color)
        
        # 木质表面属性
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)  # 半粗糙
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)   # 非金属
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04, 0.04, 0.04))
        
        # 连接材质输出
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        # 绑定材质到地面
        UsdShade.MaterialBindingAPI(ground_prim.GetPrim()).Bind(material)
        
        # 同时设置显示颜色作为备用
        ground_prim.GetDisplayColorAttr().Set([tuple(wood_color)])
    
    def _create_lighting(self):
        """创建环境光照"""
        # 环境光
        dome_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_path)
        dome_light.GetIntensityAttr().Set(500)
        dome_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
        
        # 平行光（模拟太阳）
        sun_path = "/World/SunLight"
        sun_light = UsdLux.DistantLight.Define(self.stage, sun_path)
        sun_light.GetIntensityAttr().Set(2000)
        sun_light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.85))
        
        # 设置光源方向
        sun_xform = UsdGeom.Xformable(sun_light.GetPrim())
        sun_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
        
        print("  [OK] Ambient and sun lighting created")
    
    def _create_start_goal_markers(self):
        """创建起点和终点标记"""
        # 起点标记 (绿色方块)
        start_path = "/World/StartMarker"
        start_cube = UsdGeom.Cube.Define(self.stage, start_path)
        start_cube.GetSizeAttr().Set(0.2)
        start_cube.GetDisplayColorAttr().Set([(0.2, 0.9, 0.2)])
        
        start_xform = UsdGeom.Xformable(start_cube.GetPrim())
        start_xform.AddTranslateOp().Set(Gf.Vec3d(*self.robot_pos[:2], 0.1))
        
        # 终点标记 (金色星形 - 用圆柱代替)
        goal_path = "/World/GoalMarker"
        goal_cylinder = UsdGeom.Cylinder.Define(self.stage, goal_path)
        goal_cylinder.GetRadiusAttr().Set(0.15)
        goal_cylinder.GetHeightAttr().Set(0.05)
        goal_cylinder.GetDisplayColorAttr().Set([(1.0, 0.85, 0.0)])
        
        goal_xform = UsdGeom.Xformable(goal_cylinder.GetPrim())
        goal_xform.AddTranslateOp().Set(Gf.Vec3d(*self.goal_pos[:2], 0.025))
        
        print(f"  [OK] Start marker at {self.robot_pos[:2]}")
        print(f"  [OK] Goal marker at {self.goal_pos[:2]}")
    
    def _create_dynamic_obstacles(self):
        """创建动态圆柱体障碍物"""
        # 障碍物配置：位置、半径、高度、颜色、运动类型
        obstacle_configs = [
            # 中心区域
            {'pos': [0.0, 0.0], 'radius': 0.25, 'height': 0.8, 
             'color': (0.9, 0.2, 0.2), 'motion': 'circular', 'speed': 0.5},
            # 右上区域
            {'pos': [1.2, 1.0], 'radius': 0.2, 'height': 0.6, 
             'color': (0.2, 0.5, 0.9), 'motion': 'linear_x', 'speed': 0.6},
            # 左上区域 
            {'pos': [-1.2, 1.0], 'radius': 0.3, 'height': 0.7, 
             'color': (0.2, 0.8, 0.3), 'motion': 'linear_y', 'speed': 0.55},
            # 右下区域
            {'pos': [1.0, -1.0], 'radius': 0.22, 'height': 0.5, 
             'color': (0.8, 0.5, 0.2), 'motion': 'zigzag', 'speed': 0.45},
            # 左下区域
            {'pos': [-1.0, -1.0], 'radius': 0.28, 'height': 0.65, 
             'color': (0.6, 0.2, 0.8), 'motion': 'oscillate', 'speed': 0.5},
            # 上方
            {'pos': [0.0, 1.3], 'radius': 0.24, 'height': 0.55, 
             'color': (0.9, 0.6, 0.1), 'motion': 'linear_x', 'speed': 0.7},
            # 下方
            {'pos': [0.0, -1.3], 'radius': 0.26, 'height': 0.6, 
             'color': (0.1, 0.7, 0.7), 'motion': 'linear_x', 'speed': 0.65},
            # 左侧
            {'pos': [-1.4, 0.0], 'radius': 0.23, 'height': 0.5, 
             'color': (0.7, 0.3, 0.5), 'motion': 'linear_y', 'speed': 0.6},
            # 右侧
            {'pos': [1.4, 0.0], 'radius': 0.27, 'height': 0.7, 
             'color': (0.4, 0.4, 0.9), 'motion': 'zigzag', 'speed': 0.55},
            # 对角线巡逻
            {'pos': [0.6, 0.6], 'radius': 0.21, 'height': 0.45, 
             'color': (0.8, 0.8, 0.2), 'motion': 'circular', 'speed': 0.65},
        ]
        
        for i, cfg in enumerate(obstacle_configs):
            obs_path = f"/World/Obstacle_{i}"
            cylinder = UsdGeom.Cylinder.Define(self.stage, obs_path)
            cylinder.GetRadiusAttr().Set(cfg['radius'])
            cylinder.GetHeightAttr().Set(cfg['height'])
            cylinder.GetDisplayColorAttr().Set([cfg['color']])
            
            # 设置初始位置
            xform = UsdGeom.Xformable(cylinder.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(cfg['pos'][0], cfg['pos'][1], cfg['height']/2))
            
            self.obstacles.append({
                'id': i,
                'prim_path': obs_path,
                'prim': cylinder.GetPrim(),
                'initial_pos': np.array(cfg['pos']),
                'position': np.array([cfg['pos'][0], cfg['pos'][1], cfg['height']/2]),
                'radius': cfg['radius'],
                'height': cfg['height'],
                'motion': cfg['motion'],
                'speed': cfg['speed'],
                'phase': np.random.uniform(0, 2*np.pi),  # 随机相位
                'is_dynamic': True
            })
        
        print(f"  [OK] Created {len(self.obstacles)} dynamic cylinder obstacles")
    
    def _create_robot_visual(self):
        """创建 ROSOrin 机器人 (加载 USD 模型)"""
        robot_path = "/World/Robot"
        
        # 检查 USD 文件是否存在
        if os.path.exists(self.config.usd_path):
            # 使用 USD Reference 加载完整模型
            robot_prim = self.stage.DefinePrim(robot_path)
            robot_prim.GetReferences().AddReference(self.config.usd_path)
            
            # 获取 Xformable 接口
            xform = UsdGeom.Xformable(robot_prim)
            
            # 清除现有的 xformOps（USD 模型可能自带）
            xform.ClearXformOpOrder()
            
            # 添加缩放操作 - 增大车子尺寸 (1.8倍)，使用双精度
            self.robot_scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
            self.robot_scale_op.Set(Gf.Vec3d(1.8, 1.8, 1.8))
            
            # 添加新的变换操作
            self.robot_translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            self.robot_translate_op.Set(Gf.Vec3d(self.robot_pos[0], self.robot_pos[1], 0.15))
            
            # 添加旋转操作 (用于朝向控制)
            self.robot_rotate_op = xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
            self.robot_rotate_op.Set(0.0)
            
            self.robot_prim = robot_prim
            self.use_usd_model = True
            print(f"  [OK] ROSOrin USD model loaded from: {self.config.usd_path}")
        else:
            # 回退：使用简化方块
            print(f"  [WARN] USD model not found: {self.config.usd_path}")
            print(f"         Using simplified box visual instead.")
            
            robot_box = UsdGeom.Cube.Define(self.stage, robot_path)
            robot_box.GetSizeAttr().Set(0.22)
            robot_box.GetDisplayColorAttr().Set([(0.2, 0.4, 0.9)])
            
            xform = UsdGeom.Xformable(robot_box.GetPrim())
            self.robot_translate_op = xform.AddTranslateOp()
            self.robot_translate_op.Set(Gf.Vec3d(*self.robot_pos))
            
            # 方向指示（小方块）
            arrow_path = "/World/Robot/Direction"
            arrow = UsdGeom.Cube.Define(self.stage, arrow_path)
            arrow.GetSizeAttr().Set(0.08)
            arrow.GetDisplayColorAttr().Set([(1.0, 1.0, 0.0)])
            
            arrow_xform = UsdGeom.Xformable(arrow.GetPrim())
            arrow_xform.AddTranslateOp().Set(Gf.Vec3d(0.15, 0, 0))
            
            self.robot_prim = robot_box.GetPrim()
            self.robot_rotate_op = None
            self.use_usd_model = False
            print("  [OK] Robot visual (simplified) created")
    
    def update_obstacles(self, sim_time: float):
        """更新动态障碍物位置"""
        for obs in self.obstacles:
            t = sim_time + obs['phase']
            init_pos = obs['initial_pos']
            speed = obs['speed']
            
            if obs['motion'] == 'circular':
                # 圆周运动 - 增大半径
                radius = 1.2
                x = init_pos[0] + radius * np.cos(t * speed)
                y = init_pos[1] + radius * np.sin(t * speed)
            elif obs['motion'] == 'linear_x':
                # X轴往返 - 增大幅度
                amplitude = 1.8
                x = init_pos[0] + amplitude * np.sin(t * speed)
                y = init_pos[1]
            elif obs['motion'] == 'linear_y':
                # Y轴往返 - 增大幅度
                amplitude = 1.6
                x = init_pos[0]
                y = init_pos[1] + amplitude * np.sin(t * speed)
            elif obs['motion'] == 'zigzag':
                # 之字形 - 增大幅度
                x = init_pos[0] + 1.4 * np.sin(t * speed)
                y = init_pos[1] + 0.9 * np.sin(t * speed * 2)
            elif obs['motion'] == 'oscillate':
                # 振荡 - 增大幅度
                x = init_pos[0] + 1.0 * np.sin(t * speed * 1.5)
                y = init_pos[1] + 0.7 * np.cos(t * speed)
            else:
                x, y = init_pos[0], init_pos[1]
            
            # 限制在场地内
            margin = self.arena_size / 2 - 0.5
            x = np.clip(x, -margin, margin)
            y = np.clip(y, -margin, margin)
            
            # 更新位置
            obs['position'] = np.array([x, y, obs['height']/2])
            
            # 更新 USD prim
            xform = UsdGeom.Xformable(obs['prim'])
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(x, y, obs['height']/2))
                    break
    
    def update_robot_position(self, position: np.ndarray, velocity: np.ndarray = None):
        """更新机器人位置和朝向"""
        self.robot_pos = position.copy()
        
        # 计算高度：USD 模型和简化模型高度不同
        z_height = 0.10 if hasattr(self, 'use_usd_model') and self.use_usd_model else 0.11
        
        # 确保translate op有效
        if hasattr(self, 'robot_translate_op') and self.robot_translate_op is not None:
            self.robot_translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), z_height))
        
        # 如果有速度信息，更新机器人朝向
        if velocity is not None and hasattr(self, 'robot_rotate_op') and self.robot_rotate_op is not None:
            vel_mag = np.linalg.norm(velocity)
            if vel_mag > 0.01:
                yaw = np.degrees(np.arctan2(velocity[1], velocity[0]))
                self.robot_rotate_op.Set(float(yaw))
    
    def get_visible_obstacles(self, robot_pos: np.ndarray, lidar_range: float) -> List[Dict]:
        """获取LiDAR范围内的可见障碍物"""
        visible = []
        for obs in self.obstacles:
            dist = np.linalg.norm(obs['position'][:2] - robot_pos[:2])
            if dist < lidar_range + obs['radius']:
                visible.append({
                    'id': obs['id'],
                    'position': obs['position'].copy(),
                    'radius': obs['radius'],
                    'is_dynamic': obs['is_dynamic']
                })
        return visible


# ============================================================
# 6. 可视化保存器
# ============================================================
class VisualizationSaver:
    """保存仿真可视化结果"""
    
    def __init__(self, output_dir: str, config: ROSOrinConfig):
        self.output_dir = output_dir
        self.config = config
        os.makedirs(output_dir, exist_ok=True)
        
        self.robot_history = []
        self.obstacle_history = []
        self.replan_steps = []
        
    def record_step(self, robot_pos: np.ndarray, obstacles: List[Dict], 
                   replanned: bool, step: int):
        """记录一步数据"""
        self.robot_history.append(robot_pos[:2].copy())
        self.obstacle_history.append([
            {'id': o['id'], 'pos': o['position'][:2].copy(), 'radius': o['radius']}
            for o in obstacles
        ])
        if replanned:
            self.replan_steps.append(step)
    
    def save_summary(self, goal: np.ndarray, success: bool, total_steps: int):
        """保存仿真总结图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        arena = self.config.arena_size / 2
        
        # 左图：轨迹
        ax = axes[0]
        ax.set_xlim(-arena - 0.5, arena + 0.5)
        ax.set_ylim(-arena - 0.5, arena + 0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#2c3e50')
        ax.grid(True, alpha=0.3, color='white')
        ax.set_title('ROSOrin Navigation Trajectory (Isaac Sim)', fontweight='bold', color='white')
        
        # 机器人轨迹
        robot_traj = np.array(self.robot_history)
        ax.plot(robot_traj[:, 0], robot_traj[:, 1], 'yellow', linewidth=2, label='Robot Path')
        
        # 起点终点
        ax.scatter(robot_traj[0, 0], robot_traj[0, 1], c='lime', s=150, 
                  marker='s', zorder=20, label='Start', edgecolors='white')
        ax.scatter(goal[0], goal[1], c='gold', s=200, 
                  marker='*', zorder=20, label='Goal', edgecolors='white')
        
        # 最终障碍物位置
        if self.obstacle_history:
            final_obs = self.obstacle_history[-1]
            colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']
            for i, obs in enumerate(final_obs):
                c = Circle(obs['pos'], obs['radius'], color=colors[i % len(colors)], alpha=0.7)
                ax.add_patch(c)
        
        ax.legend(loc='lower left', facecolor='#34495e', labelcolor='white')
        
        # 右图：统计
        ax2 = axes[1]
        ax2.set_facecolor('#ecf0f1')
        
        if len(self.replan_steps) > 1:
            intervals = np.diff(self.replan_steps)
            ax2.hist(intervals, bins=15, color='#3498db', alpha=0.7, edgecolor='white')
            ax2.axvline(np.mean(intervals), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(intervals):.1f}')
            ax2.legend()
        
        ax2.set_xlabel('Replan Interval (steps)')
        ax2.set_ylabel('Count')
        ax2.set_title('Replanning Statistics', fontweight='bold')
        
        # 统计信息
        final_dist = np.linalg.norm(robot_traj[-1] - goal[:2])
        stats = f"Total Steps: {total_steps}\nReplans: {len(self.replan_steps)}\n"
        stats += f"Final Dist: {final_dist:.3f}m\nSuccess: {'Yes' if success else 'No'}"
        ax2.text(0.95, 0.95, stats, transform=ax2.transAxes, va='top', ha='right',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('ROSOrin + Dual Diffusion Model - Isaac Sim Demo', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'isaac_sim_summary.png'), dpi=150)
        plt.close()
        
        print(f"\n  [OK] Summary saved to: {self.output_dir}/isaac_sim_summary.png")


# ============================================================
# 7. 主仿真循环
# ============================================================
def run_simulation():
    """运行 Isaac Sim 仿真 - 实时渲染模式"""
    print("\n" + "=" * 70)
    print("  ROSOrin + Dual Diffusion Model - Isaac Sim Navigation Demo")
    print("  - Dynamic cylinder obstacles with various motion patterns")
    print("  - Transformer-based obstacle prediction")
    print("  - Real-time smooth rendering")
    print("=" * 70)
    
    # 配置
    config = ROSOrinConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    print(f"  Seed: {SEED}")
    print(f"  Arena size: {config.arena_size}m x {config.arena_size}m")
    print(f"  Playback speed: {args.playback_speed}x")
    
    # 创建仿真上下文
    sim_cfg = SimulationCfg(
        device='cpu',
        dt=1.0/60.0,  # 60 Hz 物理仿真
        gravity=(0.0, 0.0, -9.81),
        render_interval=1,  # 每个物理步都渲染
    )
    sim = SimulationContext(sim_cfg)
    
    # 设置仿真为实时模式 (关键!)
    sim.set_setting("/app/player/useFixedTimeStepping", False)
    sim.set_setting("/app/runLoops/main/rateLimitEnabled", True)
    sim.set_setting("/app/runLoops/main/rateLimitFrequency", 60)
    
    # 获取stage
    stage = omni.usd.get_context().get_stage()
    
    # 创建场景
    scene = IsaacSimScene(stage, config)
    scene.create_scene()
    
    # 创建导航器
    navigator = DualDiffusionNavigator(device=device)
    navigator.load_models()
    
    # 创建可视化保存器
    output_dir = os.path.join(dynamic_mpd_root, 'results', 'isaac_sim_demo')
    visualizer = VisualizationSaver(output_dir, config)
    
    # 创建轨迹捕获器 (如果启用)
    trajectory_capture = None
    if args.capture:
        trajectory_capture = TrajectoryCapture(output_dir, capture_interval=args.capture_interval)
        print(f"  [OK] Trajectory capture enabled (every {args.capture_interval} steps)")
    
    # 起点和终点
    start = scene.robot_pos[:2].copy()
    goal = scene.goal_pos[:2].copy()
    
    print(f"\n  Start: {start}")
    print(f"  Goal: {goal}")
    print(f"  LiDAR range: {config.sim_lidar_range}m")
    print(f"\n  Running simulation with real-time rendering...")
    print(f"  Press Ctrl+C to stop.\n")
    
    # 重置仿真
    sim.reset()
    
    # 初始化场景渲染 - 多次更新确保稳定
    print("  Initializing renderer...")
    for _ in range(30):
        omni.kit.app.get_app().update()
    
    # 设置轨迹捕获相机
    if trajectory_capture:
        trajectory_capture.setup_camera(stage, config.arena_size)
    
    # ================================================================
    # 实时仿真循环
    # ================================================================
    robot_pos = np.array([start[0], start[1], 0.10])
    sim_time = 0.0
    max_steps = args.max_steps
    goal_reached = False
    
    # 子步数：每个逻辑步进行多次物理/渲染更新实现平滑运动
    sub_steps = 4  # 每个控制步分成4个子步
    
    try:
        for step in range(max_steps):
            # 更新障碍物位置（用于本次控制计算）
            # 注意：如果只在外层更新一次，画面会表现为“分段跳动”
            scene.update_obstacles(sim_time)
            
            # 获取可见障碍物
            visible_obs = scene.get_visible_obstacles(robot_pos, config.sim_lidar_range)
            
            # 获取控制命令
            control, replanned, reason = navigator.get_control(
                robot_pos[:2], goal, visible_obs, step
            )
            
            # 分成子步实现平滑运动
            sub_control = control / sub_steps
            
            for sub in range(sub_steps):
                # 每个物理步都更新一次障碍物，保证连续运动观感
                scene.update_obstacles(sim_time)

                # 增量更新机器人位置
                robot_pos[:2] = robot_pos[:2] + sub_control
                
                # 限制在场地内
                margin = config.arena_size / 2 - 0.3
                robot_pos[0] = np.clip(robot_pos[0], -margin, margin)
                robot_pos[1] = np.clip(robot_pos[1], -margin, margin)
                
                # 更新场景视觉
                scene.update_robot_position(robot_pos, velocity=control)
                
                # 推进物理仿真 (关键!)
                sim.step(render=True)
                
                # 更新仿真时间
                sim_time += sim_cfg.dt
            
            # 轨迹截图 (在控制步结束后)
            if trajectory_capture:
                trajectory_capture.capture_frame_simple(step, robot_pos)
            
            # 记录
            visualizer.record_step(robot_pos, visible_obs, replanned, step)
            
            # 检查到达目标
            dist_to_goal = np.linalg.norm(robot_pos[:2] - goal)
            if dist_to_goal < 0.2:
                print(f"\n  [Step {step}] GOAL REACHED! Distance: {dist_to_goal:.3f}m")
                goal_reached = True
                break
            
            # 进度报告
            if step % 50 == 0:
                print(f"  [Step {step:4d}] Pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                      f"Dist: {dist_to_goal:.2f}m")
    
    except KeyboardInterrupt:
        print("\n  Simulation interrupted by user.")
    
    # 保持最终画面 (使用物理仿真步进)
    print("\n  Holding final frame...")
    for _ in range(120):
        sim.step(render=True)
    
    # 保存结果
    total_steps = len(visualizer.robot_history)
    visualizer.save_summary(goal, goal_reached, total_steps)
    
    # 生成轨迹合成图
    if trajectory_capture:
        print("\n  Generating trajectory composite...")
        trajectory_capture.create_trajectory_composite(goal)
        trajectory_capture.create_trajectory_montage()
    
    # 统计
    final_dist = np.linalg.norm(robot_pos[:2] - goal)
    print(f"\n  Simulation Complete:")
    print(f"    Seed: {SEED}")
    print(f"    Total steps: {total_steps}")
    print(f"    Final distance: {final_dist:.3f}m")
    print(f"    Total replans: {len(visualizer.replan_steps)}")
    print(f"    Success: {'Yes' if goal_reached else 'No'}")
    print("=" * 70)
    
    # 关闭仿真
    simulation_app.close()


# ============================================================
# 8. 入口
# ============================================================
if __name__ == "__main__":
    run_simulation()

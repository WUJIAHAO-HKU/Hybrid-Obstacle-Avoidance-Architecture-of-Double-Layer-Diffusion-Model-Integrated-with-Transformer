# Dynamic MPD - 动态障碍物扩散运动规划

基于扩散模型的动态障碍物运动预测与轨迹规划系统。

---

## 🎯 项目概述

本项目提出了一种**双层扩散模型 + 斥力场引导 + 启发式评分**的混合避障架构，用于解决动态环境下的机器人路径规划问题。系统集成了激光雷达感知模块，实现了未知环境探索与实时避障。

### 核心问题

传统运动规划方法面临的挑战：

- **动态障碍物预测难**：障碍物运动模式复杂多变
- **实时性要求高**：需要快速响应环境变化
- **不确定性处理**：传感器噪声、预测误差等

### 解决方案

采用**扩散模型**的生成能力处理不确定性，结合**传统方法**的可靠性保证安全性。

---

## 🔬 技术架构与创新点

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        混合避障系统架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐                                                  │
│  │   激光雷达感知     │ ← 180射线扫描，实时检测障碍物                     │
│  │   (LidarSensor)   │   输出：障碍物位置、距离、点云                     │
│  └─────────┬─────────┘                                                  │
│            ↓                                                            │
│  ┌───────────────────┐                                                  │
│  │   局部地图构建     │ ← 障碍物历史追踪，动态/静态分类                    │
│  │   (LocalMap)      │   输出：带历史信息的障碍物列表                     │
│  └─────────┬─────────┘                                                  │
│            ↓                                                            │
│  ╔═══════════════════════════════════════════════════════════════════╗ │
│  ║                    双层扩散模型 (核心)                              ║ │
│  ╠═══════════════════════════════════════════════════════════════════╣ │
│  ║  ┌─────────────────────────────────────────────────────────────┐  ║ │
│  ║  │  上层扩散模型 (TrainableObstacleDiffusion)                   │  ║ │
│  ║  │  ─────────────────────────────────────────────────────────  │  ║ │
│  ║  │  • 输入: 障碍物历史位置 (8帧)                                │  ║ │
│  ║  │  • 输出: 未来轨迹预测 (12步)                                 │  ║ │
│  ║  │  • 扩散步数: 50                                              │  ║ │
│  ║  │  • 作用: 预测动态障碍物的未来位置分布                         │  ║ │
│  ║  └─────────────────────────────────────────────────────────────┘  ║ │
│  ║                              ↓                                     ║ │
│  ║  ┌─────────────────────────────────────────────────────────────┐  ║ │
│  ║  │  下层扩散模型 (MPD GaussianDiffusion)                        │  ║ │
│  ║  │  ─────────────────────────────────────────────────────────  │  ║ │
│  ║  │  • 输入: 起点、终点、障碍物预测位置                          │  ║ │
│  ║  │  • 输出: 避障轨迹候选集 (30条)                               │  ║ │
│  ║  │  • 扩散步数: 100                                             │  ║ │
│  ║  │  • 支撑点: 16个                                              │  ║ │
│  ║  └─────────────────────────────────────────────────────────────┘  ║ │
│  ╚═══════════════════════════════════════════════════════════════════╝ │
│            ↓                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    斥力场引导 (Repulsive Guidance)                 │ │
│  │  ─────────────────────────────────────────────────────────────── │ │
│  │  • 在扩散去噪过程中注入障碍物斥力                                  │ │
│  │  • 激活条件: t < 60 (扩散后期)                                    │ │
│  │  • 引导公式: x0_pred += guidance_scale × t_weight × repulsion     │ │
│  │  • 安全距离: obstacle_radius × 2.5                                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│            ↓                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    启发式评分选择 (Heuristic Scoring)              │ │
│  │  ─────────────────────────────────────────────────────────────── │ │
│  │  从30条候选轨迹中选择最优：                                        │ │
│  │  Score = 3.0×目标距离 + 0.3×路径长度 + 8.0×碰撞惩罚 + 0.1×转向量   │ │
│  │  • 目标距离: 终点与目标的欧氏距离                                  │ │
│  │  • 路径长度: 轨迹总长度                                           │ │
│  │  • 碰撞惩罚: 与障碍物距离<安全距离时的惩罚                         │ │
│  │  • 转向量: 轨迹平滑度指标                                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│            ↓                                                            │
│  ┌───────────────────┐                                                  │
│  │   紧急避障模块     │ ← 当障碍物距离 < 安全阈值时直接触发              │
│  │   (Emergency)     │   输出：远离障碍物的速度向量                      │
│  └───────────────────┘                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 创新点详解

### 创新点1：双层扩散模型协同

**问题**：单一扩散模型难以同时处理障碍物预测和轨迹规划。

**方案**：

- **上层模型**：专注于障碍物运动预测，学习障碍物的运动模式
- **下层模型**：基于预测结果进行轨迹规划，生成多样化候选轨迹
- **协同机制**：上层预测结果作为下层的障碍物约束条件

```python
# 上层：预测障碍物未来12步位置
predictions = upper_model.predict(obstacle_history)  # shape: (12, 2)

# 下层：在预测的障碍物分布下规划轨迹
trajectory = lower_model.plan(start, goal, predictions)
```

### 创新点2：扩散过程中的斥力场引导

**问题**：纯扩散模型生成的轨迹可能穿过障碍物。

**方案**：在扩散去噪的后期阶段（t<60）注入基于人工势场的斥力引导。

```python
# 斥力场计算
for obs_pos in obstacles:
    diff = x0_pred - obs_pos  # 轨迹点到障碍物的向量
    dist = norm(diff)
  
    if dist < safety_distance:
        repulsion = (safety_distance - dist) / safety_distance
        guidance += normalize(diff) * repulsion

# 注入引导
t_weight = (60 - t) / 60.0  # 越接近收敛，权重越大
x0_pred = x0_pred + guidance_scale * t_weight * guidance
```

**优势**：

- 保留扩散模型的多样性生成能力
- 通过斥力场保证轨迹安全性
- 权重随扩散进度自适应调整

### 创新点3：多目标启发式轨迹评分

**问题**：扩散模型生成多条候选轨迹，如何选择最优？

**方案**：综合多个目标的加权评分机制。

```python
def score_trajectory(traj, goal, obstacles):
    # 目标可达性 (权重: 3.0)
    goal_dist = distance(traj[-1], goal)
  
    # 路径效率 (权重: 0.3)
    path_length = sum(segment_lengths(traj))
  
    # 安全性 (权重: 8.0)
    collision_penalty = 0
    for obs in obstacles:
        min_dist = min_distance_to_obstacle(traj, obs)
        if min_dist < obs.radius:
            collision_penalty += 50  # 碰撞严重惩罚
        elif min_dist < safety_margin:
            collision_penalty += (safety_margin - min_dist) ** 2
  
    # 平滑度 (权重: 0.1)
    smoothness = sum(angle_changes(traj))
  
    return 3.0*goal_dist + 0.3*path_length + 8.0*collision_penalty + 0.1*smoothness
```

### 创新点4：未知环境迷雾探索

**问题**：机器人初始对环境一无所知，需要边探索边规划。

**方案**：

- **迷雾地图**：未扫描区域用半透明遮罩表示
- **增量式地图构建**：激光雷达扫描到的区域逐步揭开
- **动态重规划**：发现新障碍物时自动触发轨迹重规划

### 创新点5：动态/静态障碍物自动分类

**问题**：静态和动态障碍物需要不同的处理策略。

**方案**：基于历史位置变化自动分类。

```python
def classify_obstacle(history, threshold=0.02):
    if len(history) < 2:
        return "unknown"
  
    movement = distance(history[-1], history[0])
    return "dynamic" if movement > threshold else "static"
```

---

## 🔧 各模块作用说明

### 模块功能对照表


| 模块             | 类型   | 作用           | 输入              | 输出            |
| ---------------- | ------ | -------------- | ----------------- | --------------- |
| **上层扩散模型** | 学习型 | 预测障碍物运动 | 历史位置(8帧)     | 未来轨迹(12步)  |
| **下层扩散模型** | 学习型 | 生成候选轨迹   | 起点/终点/障碍物  | 轨迹集(30条)    |
| **斥力场引导**   | 规则型 | 保证轨迹安全   | 轨迹点/障碍物位置 | 引导向量        |
| **启发式评分**   | 规则型 | 选择最优轨迹   | 候选轨迹集        | 最佳轨迹        |
| **紧急避障**     | 规则型 | 处理紧急情况   | 当前位置/障碍物   | 逃离方向        |
| **激光雷达**     | 感知型 | 检测障碍物     | 环境              | 点云/障碍物列表 |
| **局部地图**     | 存储型 | 维护障碍物信息 | 检测结果          | 带历史的障碍物  |

### 为什么需要混合架构？


| 组件           | 扩散模型的局限           | 传统方法的补充     |
| -------------- | ------------------------ | ------------------ |
| **斥力场引导** | 可能生成穿越障碍物的轨迹 | 显式保证安全距离   |
| **启发式评分** | 无法直接优化多目标       | 综合考虑多个指标   |
| **紧急避障**   | 响应速度不够快           | 即时反应，无需规划 |

---

## 🚀 与其他方法的对比


| 方法       | 动态障碍物处理 | 不确定性建模 | 实时性    | 安全性保证      |
| ---------- | -------------- | ------------ | --------- | --------------- |
| A*/RRT     | ❌ 需要重规划  | ❌ 无        | ✅ 快     | ⚠️ 取决于采样 |
| MPC        | ⚠️ 简单预测  | ❌ 无        | ⚠️ 中等 | ✅ 约束保证     |
| DRL        | ✅ 可学习      | ⚠️ 隐式    | ✅ 快     | ❌ 无保证       |
| 纯扩散模型 | ✅ 可学习      | ✅ 显式      | ⚠️ 中等 | ❌ 无保证       |
| **本方法** | ✅ 双层预测    | ✅ 显式      | ✅ 快     | ✅ 斥力场+评分  |

---

## � 模块关系与代码结构

### 模块依赖关系图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              模块依赖关系                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         scripts/ 脚本层                                  │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  rosorin_lidar_navigation.py     ← 主演示：激光雷达导航仿真       │   │   │
│  │  │       ↓ 使用                                                     │   │   │
│  │  │  ┌───────────────────────┬───────────────────────┐              │   │   │
│  │  │  │ 上层扩散模型 (预测)    │  下层扩散模型 (规划)   │              │   │   │
│  │  │  └───────────────────────┴───────────────────────┘              │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  train_upper_diffusion_model.py  ← 训练上层模型并保存checkpoint  │   │   │
│  │  │       ↓ 使用                                                     │   │   │
│  │  │  transformer_obstacle_diffusion + complex_obstacle_data          │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  visualize_transformer_denoising.py  ← 可视化去噪过程(论文Fig)   │   │   │
│  │  │  compare_transformer_mlp.py          ← Transformer vs MLP对比    │   │   │
│  │  │  train_full_comparison.py            ← 完整对比实验               │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                         ↓ 调用                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          src/ 核心模块层                                 │   │
│  │                                                                         │   │
│  │  ┌────────────────────────────────────────────────────────────────┐    │   │
│  │  │         上层扩散模型 (障碍物轨迹预测)                            │    │   │
│  │  │  ┌─────────────────────────┬─────────────────────────────────┐ │    │   │
│  │  │  │ trainable_obstacle_     │ transformer_obstacle_           │ │    │   │
│  │  │  │ diffusion.py            │ diffusion.py                    │ │    │   │
│  │  │  │ ──────────────────────  │ ─────────────────────────────── │ │    │   │
│  │  │  │ MLP架构 (1.46M参数)     │ Transformer架构 (1.79M参数)     │ │    │   │
│  │  │  │ • 基线模型              │ • 创新模型 (93.5%改进)           │ │    │   │
│  │  │  │ • 用于对比实验          │ • 用于正式仿真和论文             │ │    │   │
│  │  │  └─────────────────────────┴─────────────────────────────────┘ │    │   │
│  │  └────────────────────────────────────────────────────────────────┘    │   │
│  │                              ↓ 依赖                                     │   │
│  │  ┌────────────────────────────────────────────────────────────────┐    │   │
│  │  │  complex_obstacle_data.py                                       │    │   │
│  │  │  ────────────────────────────────────────────────────────────  │    │   │
│  │  │  训练数据生成: 11种运动模式 (直线/曲线/圆周/避让/加减速等)       │    │   │
│  │  └────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  ┌────────────────────────────────────────────────────────────────┐    │   │
│  │  │  mpd_integration.py                                             │    │   │
│  │  │  ────────────────────────────────────────────────────────────  │    │   │
│  │  │  下层模型集成: 加载MPD原论文预训练模型，提供统一API              │    │   │
│  │  │  • load_mpd_model(): 加载预训练轨迹规划模型                     │    │   │
│  │  │  • plan_trajectory(): 调用下层扩散模型规划轨迹                  │    │   │
│  │  └────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                         ↓ 加载                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     trained_models/ 预训练权重                           │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │  best_transformer_obstacle_model.pt   ← 上层模型 (我们训练)        │ │   │
│  │  │  final_transformer_obstacle_model.pt  ← 上层模型 (最终版)          │ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                         │   │
│  │  ~/mpd-build/data_trained_models/...     ← 下层模型 (原论文预训练)     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 模块功能详解

#### 📁 src/ 核心模块


| 文件                                | 功能                        | 说明                          |
| ----------------------------------- | --------------------------- | ----------------------------- |
| `transformer_obstacle_diffusion.py` | **Transformer上层扩散模型** | ⭐ 主模型，用于正式仿真和论文 |
| `trainable_obstacle_diffusion.py`   | MLP上层扩散模型             | 基线模型，仅用于对比实验      |
| `complex_obstacle_data.py`          | 训练数据生成                | 11种运动模式                  |
| `mpd_integration.py`                | 下层模型集成                | 加载原论文预训练模型          |

#### 📁 scripts/ 脚本

**核心脚本（必须使用）：**


| 文件                                 | 用途                           | 说明                           |
| ------------------------------------ | ------------------------------ | ------------------------------ |
| `train_upper_diffusion_model.py`     | **⭐ 训练上层Transformer模型** | 首次使用前必须运行             |
| `rosorin_lidar_navigation.py`        | **⭐ 激光雷达导航仿真**        | 正式演示                       |
| `visualize_transformer_denoising.py` | **⭐ Transformer去噪可视化**   | 论文Figure，自动加载预训练模型 |

**对比实验脚本（可选）：**


| 文件                         | 用途                   | 说明     |
| ---------------------------- | ---------------------- | -------- |
| `compare_transformer_mlp.py` | Transformer vs MLP对比 | 论文实验 |
| `train_full_comparison.py`   | 完整训练对比           | 论文实验 |

**归档脚本（旧版）：**


| 文件                            | 说明                                     |
| ------------------------------- | ---------------------------------------- |
| `visualize_complex_training.py` | MLP版本可视化（已被Transformer版本替代） |
| `dual_diffusion_full.py`        | 早期系统验证                             |

---

## 🚀 工作流程

### 完整工作流

```
┌─────────────────────────────────────────────────────────────────┐
│                        工作流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: 训练上层模型 (首次运行，约10分钟)                       │
│  ─────────────────────────────────────────────                  │
│  python scripts/train_upper_diffusion_model.py                  │
│       ↓                                                         │
│  输出: trained_models/best_transformer_obstacle_model.pt        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 2: 运行仿真 (加载已训练的模型)                             │
│  ─────────────────────────────────────                          │
│  python scripts/rosorin_lidar_navigation.py                     │
│       ↓                                                         │
│  自动加载:                                                      │
│    - 上层模型: trained_models/best_transformer_obstacle_model.pt│
│    - 下层模型: ~/mpd-build/data_trained_models/...              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 3: 生成论文Figure (可选)                                  │
│  ─────────────────────────────────                              │
│  python scripts/visualize_transformer_denoising.py              │
│       ↓                                                         │
│  输出: results/transformer_denoising/*.png                      │
│  注意: 自动加载预训练模型，无需重新训练                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 快速命令

```bash
# 1. 设置环境
cd ~/mpd-build && source set_env_variables.sh
conda activate mpd-splines-public
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd "/path/to/mpd-splines-public/dynamic_mpd"

# 2. 训练上层模型 (首次运行，约10分钟)
python scripts/train_upper_diffusion_model.py --num_epochs 300 --num_train 10000

# 3. 运行仿真
python scripts/rosorin_lidar_navigation.py

# 4. 生成论文图片 (自动加载模型，无需训练)
python scripts/visualize_transformer_denoising.py

# 5. 运行Isaac Sim 3D仿真 (需要安装Isaac Sim)
cd "/home/wujiahao/ROSORIN_CAR and Reasearch/Diffusion-Guided Path Optimization (DGPO)"
./isaaclab_runner.sh "/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public/dynamic_mpd/scripts/rosorin_demo/isaac_sim_navigation.py" 2>&1
```

---

## 🤖 ROSOrin机器人演示

本项目支持在ROSOrin麦克纳姆轮小车上部署和验证双层扩散模型。

### ROSOrin机器人参数


| 参数       | 值                             |
| ---------- | ------------------------------ |
| 机器人类型 | 麦克纳姆轮全向移动平台         |
| 底盘尺寸   | 轮距 206mm                     |
| LiDAR      | MS200激光雷达 (360°, 12m范围) |
| 深度相机   | Aurora930                      |
| 主控       | Jetson Orin                    |
| 最大速度   | 0.5 m/s                        |

### 演示脚本说明

#### 1. 纯Python仿真 (`scripts/rosorin_lidar_navigation.py`)

使用matplotlib进行可视化的纯Python仿真，不需要Isaac Sim。

**功能特点：**

- 实时LiDAR模拟（180射线，12米范围）
- 动态障碍物（圆周运动、直线运动）
- Transformer上层扩散模型预测障碍物轨迹
- 势场法+扩散预测融合的轨迹规划
- 迷雾地图探索可视化

**运行方式：**

```bash
cd ~/mpd-build && source set_env_variables.sh
conda activate mpd-splines-public
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/mpd-build/dynamic_mpd
python scripts/rosorin_lidar_navigation.py
```

#### 2. Isaac Sim 3D仿真 (`scripts/rosorin_demo/isaac_sim_navigation.py`)

在NVIDIA Isaac Sim中运行的完整3D仿真，需要安装Isaac Sim和Isaac Lab。

**功能特点：**

- 3D物理仿真环境（带环境光照）
- USD场景创建（地面、围墙、动态圆柱体障碍物）
- 机器人从一边穿越到另一边的导航任务
- 实时动态障碍物更新
- Transformer扩散模型集成
- 可视化帧自动保存

**运行方式：**

```bash
# 使用 DGPO 项目的 isaaclab_runner.sh 启动
cd "/home/wujiahao/ROSORIN_CAR and Reasearch/Diffusion-Guided Path Optimization (DGPO)"
./isaaclab_runner.sh "/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public/dynamic_mpd/scripts/rosorin_demo/isaac_sim_navigation.py"
```

**可选参数：**

```bash
--headless  # 无头模式运行（不显示GUI，适合服务器）
--num_envs N  # 并行环境数量
```

### 结果保存位置


| 仿真类型      | 输出目录                    |
| ------------- | --------------------------- |
| 纯Python仿真  | `results/lidar_navigation/` |
| Isaac Sim仿真 | `results/isaac_sim_demo/`   |

---

## 项目结构

```
dynamic_mpd/
├── src/                                  # 核心模块
│   ├── transformer_obstacle_diffusion.py # ⭐ 上层扩散模型 (Transformer, 主模型)
│   ├── trainable_obstacle_diffusion.py   # 上层扩散模型 (MLP, 基线对比)
│   ├── complex_obstacle_data.py          # 训练数据生成 (11种运动模式)
│   └── mpd_integration.py                # 下层MPD模型集成
│
├── scripts/                              # 脚本
│   ├── train_upper_diffusion_model.py    # ⭐ [Step 1] 训练上层模型
│   ├── rosorin_lidar_navigation.py       # ⭐ [Step 2] 激光雷达导航仿真
│   ├── visualize_transformer_denoising.py# ⭐ [Step 3] 论文Figure生成
│   ├── compare_transformer_mlp.py        # (可选) Transformer vs MLP对比
│   ├── train_full_comparison.py          # (可选) 完整对比实验
│   ├── visualize_complex_training.py     # (归档) MLP版本可视化
│   └── rosorin_demo/                     # ⭐ ROSOrin机器人演示
│       ├── isaac_sim_navigation.py       # Isaac Sim 3D导航仿真
│       └── rosorin_lidar_navigation.py   # 纯Python 2D导航仿真
│
├── trained_models/                       # ⭐ 训练好的模型权重
│   ├── best_transformer_obstacle_model.pt   # 最佳上层模型
│   └── final_transformer_obstacle_model.pt  # 最终上层模型
│
├── results/                              # 输出结果
│   ├── transformer_denoising/            # 论文Figure输出
│   ├── lidar_navigation/                 # 仿真结果
│   └── isaac_sim_demo/                   # Isaac Sim演示帧
│
└── README.md
```

---

## ⚠️ 重要：mpd-build 与预训练模型

### 项目目录关系

本项目依赖原论文 MPD 的预训练模型。目前有**两个相关目录**：


| 目录                   | 路径           | 说明                                     |
| ---------------------- | -------------- | ---------------------------------------- |
| **mpd-build**          | `~/mpd-build/` | 原论文项目的编译安装目录，包含预训练模型 |
| **mpd-splines-public** | 当前项目父目录 | 我们的动态避障扩展项目                   |

### 预训练模型位置

**下层扩散模型（轨迹规划）- 原论文预训练：**

```bash
# 软链接结构
~/mpd-build/data_trained_models -> ~/mpd-build/data_public/data_trained_models

# 实际模型路径
~/mpd-build/data_public/data_trained_models/
├── launch_train_diffusion_models-v04_2024-09-17_21-44-17/
│   └── generative_model_class___GaussianDiffusionModel/
│       ├── dataset_subdir___EnvSimple2D-RobotPointMass2D.../
│       │   └── .../checkpoints/
│       │       ├── model_current.pth          # 下层扩散模型
│       │       └── ema_model_current.pth      # EMA版本 (推荐使用)
│       ├── dataset_subdir___EnvDense2D-RobotPointMass2D.../
│       └── dataset_subdir___EnvNarrowPassageDense2D.../
└── ...
```

**上层扩散模型（障碍物预测）- 我们训练的：**

```bash
# 我们训练的障碍物预测模型
dynamic_mpd/old_results/trained_diffusion_model.pth
```

### 配置文件路径说明

脚本中的配置文件位于：

```bash
~/mpd-build/scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml
```

配置文件中使用 `${HOME}/mpd-build/...` 作为模型路径前缀。

### 运行前环境设置

```bash
# 1. 进入 mpd-build 设置环境变量
cd ~/mpd-build
source set_env_variables.sh

# 2. 激活 conda 环境
conda activate mpd-splines-public

# 3. 设置动态库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. 运行脚本
cd "/path/to/mpd-splines-public/dynamic_mpd"
python scripts/rosorin_lidar_navigation.py
```

### 模型使用总结


| 模型层级       | 功能           | 架构        | 来源         | 路径                                                        |
| -------------- | -------------- | ----------- | ------------ | ----------------------------------------------------------- |
| **上层**       | 障碍物轨迹预测 | Transformer | 我们训练     | `trained_models/best_transformer_obstacle_model.pt`         |
| **上层(基线)** | 障碍物轨迹预测 | MLP         | 我们训练     | `old_results/trained_diffusion_model.pth`                   |
| **下层**       | 机器人轨迹规划 | UNet        | 原论文预训练 | `~/mpd-build/data_trained_models/.../ema_model_current.pth` |

### 快速开始

#### 方式1：2D LiDAR 仿真（纯Python，推荐快速验证）

```bash
# 1. 设置环境
cd ~/mpd-build && source set_env_variables.sh
conda activate mpd-splines-public
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 2. 进入项目目录
cd ~/mpd-build/dynamic_mpd

# 3. 训练上层模型 (首次运行，约30分钟)
python scripts/train_upper_diffusion_model.py --num_epochs 300 --num_train 10000

# 4. 运行2D仿真（带LiDAR可视化）
python scripts/rosorin_lidar_navigation.py

# 5. 生成论文图片
python scripts/visualize_transformer_denoising.py
```

#### 方式2：Isaac Sim 3D 仿真（需要安装 Isaac Lab）

```bash
# 1. 进入 DGPO 项目目录（包含 isaaclab_runner.sh）
cd "/home/wujiahao/ROSORIN_CAR and Reasearch/Diffusion-Guided Path Optimization (DGPO)"

# 2. 运行 Isaac Sim 导航演示
./isaaclab_runner.sh "/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public/dynamic_mpd/scripts/rosorin_demo/isaac_sim_navigation.py"

# 可选参数：
#   --headless      无界面模式运行
#   --max_steps 800 设置最大仿真步数
```

**Isaac Sim 场景说明：**

- 场地：4m × 4m 平面，带环境光和平行光
- 机器人：ROSOrin 麦克纳姆轮小车（简化为蓝色方块）
- 障碍物：5个动态移动圆柱体，各有不同运动模式：
  - `circular`: 圆周运动
  - `linear_x/y`: X/Y轴往返运动
  - `zigzag`: 之字形运动
  - `oscillate`: 振荡运动
- 任务：从左下角穿越到右上角，避开所有动态障碍物

### 加载预训练模型 (Python API)

```python
from src.transformer_obstacle_diffusion import TransformerObstacleDiffusionWrapper

# 方式1: 从checkpoint加载
model = TransformerObstacleDiffusionWrapper.load_from_checkpoint(
    'trained_models/best_transformer_obstacle_model.pt',
    device='cuda'
)

# 方式2: 先创建再加载
model = TransformerObstacleDiffusionWrapper(device='cuda')
model.load('trained_models/best_transformer_obstacle_model.pt')

# 预测障碍物未来轨迹
# history: [8, 2] 过去8帧位置
mean_pred, std_pred, samples = model.predict(history, num_samples=20)
# mean_pred: [12, 2] 未来12步预测均值
```

---

### 预训练模型位置（本地副本）

如果 `~/mpd-build` 不可用，可使用当前项目下的副本：

```
mpd-splines-public/data_public/data_trained_models/
├── launch_train_diffusion_models-v04_2024-09-17_21-44-17/
│   └── generative_model_class___GaussianDiffusionModel/
│       ├── dataset_subdir___EnvSimple2D-RobotPointMass2D.../
│       │   └── checkpoints/
│       │       ├── model_current.pth          # 下层扩散模型
│       │       └── ema_model_current.pth      # EMA版本
│       ├── dataset_subdir___EnvDense2D-RobotPointMass2D.../
│       ├── dataset_subdir___EnvNarrowPassageDense2D.../
│       └── dataset_subdir___EnvPlanar2Link.../
└── ...
```


| 环境                    | 用途         | 模型路径                                                           |
| ----------------------- | ------------ | ------------------------------------------------------------------ |
| EnvSimple2D             | 简单2D障碍物 | `.../EnvSimple2D.../checkpoints/ema_model_current.pth`             |
| EnvDense2D              | 密集障碍物   | `.../EnvDense2D.../checkpoints/ema_model_current.pth`              |
| EnvNarrowPassageDense2D | 狭窄通道     | `.../EnvNarrowPassageDense2D.../checkpoints/ema_model_current.pth` |

---

## 📊 实验结果 (用于论文)

### 实验配置


| 配置项                | 值                                             |
| --------------------- | ---------------------------------------------- |
| 训练样本数            | 10,000                                         |
| 测试样本数            | 2,000                                          |
| 训练轮数 (Epochs)     | 200                                            |
| 批量大小 (Batch Size) | 64                                             |
| 学习率                | 1e-4                                           |
| 观测历史长度          | 8 帧                                           |
| 预测时域              | 12 步                                          |
| 状态维度              | 2 (x, y)                                       |
| 扩散步数              | 50                                             |
| 运动类型              | circular, linear, zigzag, spiral, acceleration |
| GPU                   | NVIDIA RTX (CUDA)                              |
| 随机种子              | 42                                             |

### 主要结果对比


| 模型                   | Final ADE ↓   | Final FDE ↓   | Best ADE ↓    | 训练时间 |
| ---------------------- | -------------- | -------------- | -------------- | -------- |
| MLP Baseline           | 1.1250         | 1.1300         | 0.6346         | ~12 min  |
| **Transformer (Ours)** | **0.0726**     | **0.1534**     | **0.0661**     | ~11 min  |
| **改进幅度**           | **93.5%** ⬆️ | **86.4%** ⬆️ | **89.6%** ⬆️ | -        |

> **ADE** (Average Displacement Error): 平均位移误差，所有预测点与真实值的平均欧氏距离
> **FDE** (Final Displacement Error): 最终位移误差，预测终点与真实终点的欧氏距离

### 训练收敛曲线

```
训练损失变化:
                                                                
  MLP Baseline:     ████████████████████████████████ 1.0057 (epoch 200)
  Transformer:      █ 0.0384 (epoch 200)
              
  结论: Transformer在相同epochs下收敛到更低的损失
```

### 各运动类型性能分析


| 运动类型            | MLP ADE | Transformer ADE | 改进  |
| ------------------- | ------- | --------------- | ----- |
| Circular (圆周)     | ~1.2    | ~0.08           | 93.3% |
| Linear (直线)       | ~0.9    | ~0.05           | 94.4% |
| Zigzag (之字形)     | ~1.3    | ~0.09           | 93.1% |
| Spiral (螺旋)       | ~1.1    | ~0.07           | 93.6% |
| Acceleration (变速) | ~1.0    | ~0.08           | 92.0% |

### 模型参数对比


| 参数项     | MLP Baseline | Transformer     |
| ---------- | ------------ | --------------- |
| 总参数量   | 1,462,808    | 1,788,546       |
| 参数增加   | -            | +22.3%          |
| 隐藏维度   | 256          | 128 (d_model)   |
| 网络层数   | 6 (FC)       | 4 (Transformer) |
| 注意力头数 | -            | 4               |
| FFN维度    | -            | 512             |
| Dropout    | 0.1          | 0.1             |

### Transformer架构优势分析


| 组件                             | 作用                     | 对性能的贡献             |
| -------------------------------- | ------------------------ | ------------------------ |
| **Temporal Self-Attention**      | 捕捉历史轨迹中的时序依赖 | 理解运动趋势和加速度变化 |
| **Adaptive LayerNorm**           | 将扩散时间步信息注入网络 | 更有效的条件去噪         |
| **Sinusoidal Position Encoding** | 编码序列位置信息         | 泛化到不同长度的输入     |
| **Cross-Attention** (可选)       | 建模多障碍物之间的交互   | 处理复杂多目标场景       |

### 可视化结果

实验结果保存在 `results/full_comparison/`:


| 文件                            | 内容                            |
| ------------------------------- | ------------------------------- |
| `training_curves.png`           | MLP vs Transformer 训练损失曲线 |
| `prediction_by_motion_type.png` | 各运动类型预测可视化            |
| `results_table.png`             | 实验结果汇总表格                |

### 复现命令

```bash
# 激活环境
cd ~/mpd-build && source set_env_variables.sh
conda activate mpd-splines-public
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 运行完整对比实验
cd "/path/to/mpd-splines-public/dynamic_mpd"
python scripts/train_full_comparison.py
```

### 论文表格格式 (LaTeX)

```latex
\begin{table}[h]
\centering
\caption{Obstacle Trajectory Prediction: MLP vs Transformer}
\label{tab:comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{ADE} $\downarrow$ & \textbf{FDE} $\downarrow$ & \textbf{Params} \\
\midrule
MLP Baseline & 1.1250 & 1.1300 & 1.46M \\
Transformer (Ours) & \textbf{0.0726} & \textbf{0.1534} & 1.79M \\
\midrule
Improvement & 93.5\% & 86.4\% & +22.3\% \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 🧠 Transformer扩散模型创新点

### 架构设计 (`src/transformer_obstacle_diffusion.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│               Transformer Obstacle Diffusion                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Temporal Transformer Encoder                             │  │
│  │  • Input Embedding + Sinusoidal Position Encoding         │  │
│  │  • Self-Attention: 捕捉历史轨迹的时序依赖                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Cross-Attention: 建模多障碍物交互关系                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Diffusion Transformer Decoder                            │  │
│  │  • Adaptive LayerNorm: 注入扩散时间步信息                 │  │
│  │  • Output: pred_horizon × state_dim                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心创新


| 创新点                      | 技术细节                 | 相比MLP的优势          |
| --------------------------- | ------------------------ | ---------------------- |
| **Temporal Self-Attention** | 多头自注意力捕捉时序依赖 | 不受固定窗口限制       |
| **Cross-Attention**         | 建模多障碍物交互         | 动态感知场景变化       |
| **Adaptive LayerNorm**      | 时间步调制归一化参数     | 比拼接更有效的条件注入 |
| **Sinusoidal Position**     | 连续位置编码             | 泛化到不同长度序列     |

---

### 论文贡献点总结

1. **双层扩散架构**: 上层预测障碍物运动，下层规划避障轨迹
2. **Transformer障碍物预测**: 相比MLP提升93.5% (ADE)
3. **斥力场引导采样**: 在扩散过程中注入物理约束
4. **混合评分机制**: 多目标启发式轨迹选择
5. **未知环境探索**: 激光雷达感知 + 迷雾地图

### 实验对比基准 (Baselines)


| 基准方法           | 描述                 | 对比维度       |
| ------------------ | -------------------- | -------------- |
| MLP Diffusion      | 全连接网络作为去噪器 | 预测精度       |
| Linear Predictor   | 线性外推预测         | 动态障碍物处理 |
| LSTM/GRU           | 循环神经网络预测     | 时序建模能力   |
| Social Force Model | 社会力模型           | 多障碍物交互   |
| MPD (原论文)       | 静态环境扩散规划     | 动态适应性     |

### 关键引用

```bibtex
@article{carvalho2025motion,
  title={Motion planning diffusion: Learning and adapting robot motion planning with diffusion models},
  author={Carvalho, Jo{\~a}o and Le, An T and Kicki, Piotr and Koert, Dorothea and Peters, Jan},
  journal={IEEE Transactions on Robotics},
  year={2025},
  publisher={IEEE}
}

@inproceedings{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}
```

---

## 🔄 更新日志


| 日期       | 版本   | 更新内容                                  |
| ---------- | ------ | ----------------------------------------- |
| 2026-01-29 | v0.3.0 | 添加Transformer上层扩散模型，完成对比实验 |
| 2026-01-28 | v0.2.5 | 优化终点附近避障策略                      |
| 2026-01-27 | v0.2.0 | 整理代码结构，归档旧脚本                  |
| 2026-01-22 | v0.1.0 | 初始双层扩散架构实现                      |

---

## 🆕 可视化效果说明

### 迷雾地图 (Fog of War)


| 区域             | 颜色         | 说明                     |
| ---------------- | ------------ | ------------------------ |
| 未探索           | 深灰色半透明 | 机器人未扫描到的区域     |
| 已探索           | 浅绿色渐变   | 激光雷达扫描过的安全区域 |
| 障碍物（检测中） | 高亮彩色     | 当前LiDAR范围内的障碍物  |
| 障碍物（历史）   | 淡色         | 之前发现但当前不在范围内 |

### 激光线颜色


| 颜色 | 含义                     |
| ---- | ------------------------ |
| 青色 | 射线未碰到障碍物（空闲） |
| 红色 | 射线碰到障碍物（检测）   |

### 扩散收敛可视化


| 颜色     | 含义                 |
| -------- | -------------------- |
| 红色轨迹 | 初始噪声状态 (t=100) |
| 蓝色轨迹 | 收敛后状态 (t=0)     |
| 绿色轨迹 | 最佳选择轨迹         |

---

## 注意事项

1. **预训练模型**: 下层扩散模型使用原始论文预训练权重（位于 `data_trained_models/`）
2. **显存需求**: GPU训练建议 ≥4GB
3. **随机种子**: 所有脚本使用 `SEED=42` 确保可复现
4. **依赖项**: `pip install torch numpy matplotlib scipy pillow`

---

## License

MIT License

## Author

Dynamic MPD Project

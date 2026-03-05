# Dynamic MPD - Dynamic Obstacle Avoidance via Diffusion-Based Motion Planning

A diffusion model-based system for dynamic obstacle motion prediction and trajectory planning.

---

## 🎯 Project Overview

This project presents a **hybrid obstacle avoidance architecture** combining dual-layer diffusion models, repulsive guidance fields, and heuristic scoring for robot path planning in dynamic environments. The system integrates LiDAR perception modules to enable unknown environment exploration and real-time obstacle avoidance.

### Core Challenges

Traditional motion planning methods face:

- **Dynamic Obstacle Prediction**: Complex and variable obstacle motion patterns
- **Real-time Requirements**: Need for rapid response to environmental changes
- **Uncertainty Handling**: Sensor noise, prediction errors, etc.

### Solution

Leveraging the **generative capabilities of diffusion models** to handle uncertainty while combining **traditional methods' reliability** to ensure safety.

---

## 🔬 Technical Architecture & Innovations

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Hybrid Obstacle Avoidance System                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐                                                  │
│  │   LiDAR Sensing   │ ← 180-ray scan, real-time obstacle detection     │
│  │   (LidarSensor)   │   Output: obstacle positions, distances, point   │
│  └─────────┬─────────┘   cloud                                          │
│            ↓                                                            │
│  ┌───────────────────┐                                                  │
│  │   Local Mapping   │ ← Obstacle history tracking, dynamic/static      │
│  │   (LocalMap)      │   classification                                 │
│  └─────────┬─────────┘   Output: obstacle list with history             │
│            ↓                                                            │
│  ╔═══════════════════════════════════════════════════════════════════╗ │
│  ║              Dual-Layer Diffusion Model (Core)                     ║ │
│  ╠═══════════════════════════════════════════════════════════════════╣ │
│  ║  ┌─────────────────────────────────────────────────────────────┐  ║ │
│  ║  │  Upper Diffusion Model (TrainableObstacleDiffusion)         │  ║ │
│  ║  │  ─────────────────────────────────────────────────────────  │  ║ │
│  ║  │  • Input: Obstacle history positions (8 frames)             │  ║ │
│  ║  │  • Output: Future trajectory prediction (12 steps)          │  ║ │
│  ║  │  • Diffusion steps: 50                                      │  ║ │
│  ║  │  • Function: Predict future position distribution           │  ║ │
│  ║  └─────────────────────────────────────────────────────────────┘  ║ │
│  ║                              ↓                                     ║ │
│  ║  ┌─────────────────────────────────────────────────────────────┐  ║ │
│  ║  │  Lower Diffusion Model (MPD GaussianDiffusion)              │  ║ │
│  ║  │  ─────────────────────────────────────────────────────────  │  ║ │
│  ║  │  • Input: Start, goal, predicted obstacle positions         │  ║ │
│  ║  │  • Output: Candidate trajectory set (30 trajectories)       │  ║ │
│  ║  │  • Diffusion steps: 100                                     │  ║ │
│  ║  │  • Waypoints: 16 control points                             │  ║ │
│  ║  └─────────────────────────────────────────────────────────────┘  ║ │
│  ╚═══════════════════════════════════════════════════════════════════╝ │
│            ↓                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                 Repulsive Field Guidance                          │ │
│  │  ─────────────────────────────────────────────────────────────── │ │
│  │  • Inject obstacle repulsion during diffusion denoising          │ │
│  │  • Activation: t < 60 (late diffusion stage)                     │ │
│  │  • Guidance formula: x0_pred += scale × t_weight × repulsion     │ │
│  │  • Safety distance: obstacle_radius × 2.5                        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│            ↓                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                 Heuristic Trajectory Scoring                      │ │
│  │  ─────────────────────────────────────────────────────────────── │ │
│  │  Select optimal from 30 candidate trajectories:                  │ │
│  │  Score = 3.0×goal_dist + 0.3×path_len + 8.0×collision + 0.1×turn │ │
│  │  • Goal distance: Euclidean distance to target                   │ │
│  │  • Path length: Total trajectory length                          │ │
│  │  • Collision penalty: Penalty when distance < safety margin      │ │
│  │  • Turning: Trajectory smoothness metric                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│            ↓                                                            │
│  ┌───────────────────┐                                                  │
│  │ Emergency Avoid   │ ← Triggered when obstacle distance < threshold   │
│  │   (Emergency)     │   Output: Velocity vector away from obstacle     │
│  └───────────────────┘                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 Innovation Details

### Innovation 1: Dual-Layer Diffusion Collaboration

**Problem**: A single diffusion model struggles to handle both obstacle prediction and trajectory planning simultaneously.

**Solution**:

- **Upper Model**: Focuses on obstacle motion prediction, learning obstacle movement patterns
- **Lower Model**: Plans collision-free trajectories based on predictions, generating diverse candidates
- **Collaboration**: Upper predictions serve as obstacle constraints for lower model

```python
# Upper layer: Predict obstacle positions for next 12 steps
predictions = upper_model.predict(obstacle_history)  # shape: (12, 2)

# Lower layer: Plan trajectory under predicted obstacle distribution
trajectory = lower_model.plan(start, goal, predictions)
```

### Innovation 2: Repulsive Field Guidance in Diffusion

**Problem**: Pure diffusion models may generate trajectories that pass through obstacles.

**Solution**: Inject artificial potential field-based repulsion guidance in late diffusion stages (t<60).

```python
# Compute repulsive forces
for obs_pos in obstacles:
    diff = x0_pred - obs_pos  # Vector from trajectory point to obstacle
    dist = norm(diff)
  
    if dist < safety_distance:
        repulsion = (safety_distance - dist) / safety_distance
        guidance += normalize(diff) * repulsion

# Inject guidance
t_weight = (60 - t) / 60.0  # Higher weight closer to convergence
x0_pred = x0_pred + guidance_scale * t_weight * guidance
```

**Advantages**:

- Preserves diffusion model's diversity generation capability
- Ensures trajectory safety through repulsive fields
- Weight adapts with diffusion progress

### Innovation 3: Multi-Objective Heuristic Trajectory Scoring

**Problem**: How to select the optimal trajectory from multiple diffusion-generated candidates?

**Solution**: Weighted scoring mechanism combining multiple objectives.

```python
def score_trajectory(traj, goal, obstacles):
    # Goal reachability (weight: 3.0)
    goal_dist = distance(traj[-1], goal)
  
    # Path efficiency (weight: 0.3)
    path_length = sum(segment_lengths(traj))
  
    # Safety (weight: 8.0)
    collision_penalty = 0
    for obs in obstacles:
        min_dist = min_distance_to_obstacle(traj, obs)
        if min_dist < obs.radius:
            collision_penalty += 50  # Severe collision penalty
        elif min_dist < safety_margin:
            collision_penalty += (safety_margin - min_dist) ** 2
  
    # Smoothness (weight: 0.1)
    smoothness = sum(angle_changes(traj))
  
    return 3.0*goal_dist + 0.3*path_length + 8.0*collision_penalty + 0.1*smoothness
```

### Innovation 4: Unknown Environment Fog Exploration

**Problem**: Robot initially has no knowledge of the environment and needs to explore while planning.

**Solution**:

- **Fog Map**: Unscanned regions represented with semi-transparent overlay
- **Incremental Map Building**: LiDAR-scanned regions progressively revealed
- **Dynamic Replanning**: Automatic trajectory replanning upon discovering new obstacles

### Innovation 5: Automatic Dynamic/Static Obstacle Classification

**Problem**: Static and dynamic obstacles require different handling strategies.

**Solution**: Automatic classification based on historical position changes.

```python
def classify_obstacle(history, threshold=0.02):
    if len(history) < 2:
        return "unknown"
  
    movement = distance(history[-1], history[0])
    return "dynamic" if movement > threshold else "static"
```

---

## 🔧 Module Function Overview

### Module Function Table


| Module                  | Type     | Function                  | Input                  | Output                    |
| ----------------------- | -------- | ------------------------- | ---------------------- | ------------------------- |
| **Upper Diffusion**     | Learning | Predict obstacle motion   | History positions (8)  | Future trajectory (12)    |
| **Lower Diffusion**     | Learning | Generate candidate paths  | Start/goal/obstacles   | Trajectory set (30)       |
| **Repulsive Guidance**  | Rule     | Ensure trajectory safety  | Waypoints/obstacle pos | Guidance vectors          |
| **Heuristic Scoring**   | Rule     | Select optimal trajectory | Candidate trajectories | Best trajectory           |
| **Emergency Avoidance** | Rule     | Handle emergencies        | Current pos/obstacles  | Escape direction          |
| **LiDAR**               | Sensing  | Detect obstacles          | Environment            | Point cloud/obstacle list |
| **Local Map**           | Storage  | Maintain obstacle info    | Detection results      | Obstacles with history    |

### Why Hybrid Architecture?


| Component               | Diffusion Limitation                 | Traditional Method Supplement   |
| ----------------------- | ------------------------------------ | ------------------------------- |
| **Repulsive Guidance**  | May generate obstacle-crossing paths | Explicitly guarantee safety     |
| **Heuristic Scoring**   | Cannot directly optimize multi-goals | Comprehensive metric evaluation |
| **Emergency Avoidance** | Insufficient response speed          | Instant reaction, no planning   |

---

## 🚀 Comparison with Other Methods


| Method         | Dynamic Handling | Uncertainty   | Real-time   | Safety Guarantee    |
| -------------- | ---------------- | ------------- | ----------- | ------------------- |
| A*/RRT         | ❌ Replanning    | ❌ None       | ✅ Fast     | ⚠️ Sampling-based |
| MPC            | ⚠️ Simple pred | ❌ None       | ⚠️ Medium | ✅ Constraint-based |
| DRL            | ✅ Learnable     | ⚠️ Implicit | ✅ Fast     | ❌ No guarantee     |
| Pure Diffusion | ✅ Learnable     | ✅ Explicit   | ⚠️ Medium | ❌ No guarantee     |
| **Our Method** | ✅ Dual predict  | ✅ Explicit   | ✅ Fast     | ✅ Repulsion+score  |

---

## 📦 Module Dependencies & Code Structure

### Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Module Dependencies                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         scripts/ Layer                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  rosorin_lidar_navigation.py     ← Main demo: LiDAR navigation   │   │   │
│  │  │       ↓ uses                                                     │   │   │
│  │  │  ┌───────────────────────┬───────────────────────┐              │   │   │
│  │  │  │ Upper Diffusion (pred)│  Lower Diffusion (plan)│              │   │   │
│  │  │  └───────────────────────┴───────────────────────┘              │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  train_upper_diffusion_model.py  ← Train upper model checkpoint  │   │   │
│  │  │       ↓ uses                                                     │   │   │
│  │  │  transformer_obstacle_diffusion + complex_obstacle_data          │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  visualize_transformer_denoising.py  ← Visualize for paper      │   │   │
│  │  │  compare_transformer_mlp.py          ← Transformer vs MLP        │   │   │
│  │  │  train_full_comparison.py            ← Full comparison exp       │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                         ↓ calls                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          src/ Core Module Layer                          │   │
│  │                                                                         │   │
│  │  ┌────────────────────────────────────────────────────────────────┐    │   │
│  │  │        Upper Diffusion Model (Obstacle Prediction)              │    │   │
│  │  │  ┌─────────────────────────┬─────────────────────────────────┐ │    │   │
│  │  │  │ trainable_obstacle_     │ transformer_obstacle_           │ │    │   │
│  │  │  │ diffusion.py            │ diffusion.py                    │ │    │   │
│  │  │  │ ──────────────────────  │ ─────────────────────────────── │ │    │   │
│  │  │  │ MLP architecture (1.46M)│ Transformer arch (1.79M params) │ │    │   │
│  │  │  │ • Baseline model        │ • Novel model (93.5% improve)   │ │    │   │
│  │  │  │ • For comparison        │ • For official sim & paper      │ │    │   │
│  │  │  └─────────────────────────┴─────────────────────────────────┘ │    │   │
│  │  └────────────────────────────────────────────────────────────────┘    │   │
│  │                              ↓ depends                                  │   │
│  │  ┌────────────────────────────────────────────────────────────────┐    │   │
│  │  │  complex_obstacle_data.py                                       │    │   │
│  │  │  ────────────────────────────────────────────────────────────  │    │   │
│  │  │  Training data generation: 11 motion patterns                   │    │   │
│  │  │  (linear/curve/circular/evasion/accel-decel etc.)               │    │   │
│  │  └────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                         │   │
│  │  ┌────────────────────────────────────────────────────────────────┐    │   │
│  │  │  mpd_integration.py                                             │    │   │
│  │  │  ────────────────────────────────────────────────────────────  │    │   │
│  │  │  Lower model integration: Load original MPD pretrained model    │    │   │
│  │  │  • load_mpd_model(): Load pretrained trajectory planner         │    │   │
│  │  │  • plan_trajectory(): Call lower diffusion model for planning   │    │   │
│  │  └────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                         ↓ loads                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     trained_models/ Pretrained Weights                   │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │   │
│  │  │  best_transformer_obstacle_model.pt   ← Upper model (ours)        │ │   │
│  │  │  final_transformer_obstacle_model.pt  ← Upper model (final)       │ │   │
│  │  └───────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                         │   │
│  │  ~/mpd-build/data_trained_models/...     ← Lower model (original MPD)  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Module Details

#### 📁 src/ Core Modules


| File                                | Function                        | Note                                   |
| ----------------------------------- | ------------------------------- | -------------------------------------- |
| `transformer_obstacle_diffusion.py` | **Transformer upper diffusion** | ⭐ Main model for official sim & paper |
| `trainable_obstacle_diffusion.py`   | MLP upper diffusion             | Baseline for comparison only           |
| `complex_obstacle_data.py`          | Training data generation        | 11 motion patterns                     |
| `mpd_integration.py`                | Lower model integration         | Load original MPD pretrained model     |

#### 📁 scripts/ Scripts

**Core Scripts (Must Use):**


| File                                 | Purpose                              | Note                                |
| ------------------------------------ | ------------------------------------ | ----------------------------------- |
| `train_upper_diffusion_model.py`     | **⭐ Train upper Transformer model** | Must run before first use           |
| `rosorin_lidar_navigation.py`        | **⭐ LiDAR navigation simulation**   | Official demo                       |
| `visualize_transformer_denoising.py` | **⭐ Transformer denoising viz**     | Paper figures, auto-load pretrained |

**Comparison Experiment Scripts (Optional):**


| File                         | Purpose                       | Note      |
| ---------------------------- | ----------------------------- | --------- |
| `compare_transformer_mlp.py` | Transformer vs MLP comparison | Paper exp |
| `train_full_comparison.py`   | Full training comparison      | Paper exp |

**Archived Scripts (Legacy):**


| File                            | Note                                      |
| ------------------------------- | ----------------------------------------- |
| `visualize_complex_training.py` | MLP version viz (replaced by Transformer) |
| `dual_diffusion_full.py`        | Early system validation                   |

---

## 🚀 Workflow

### Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Workflow                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Train upper model (first run, ~10 minutes)             │
│  ─────────────────────────────────────────────                  │
│  python scripts/train_upper_diffusion_model.py                  │
│       ↓                                                         │
│  Output: trained_models/best_transformer_obstacle_model.pt      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 2: Run simulation (load trained models)                   │
│  ─────────────────────────────────────                          │
│  python scripts/rosorin_lidar_navigation.py                     │
│       ↓                                                         │
│  Auto-load:                                                     │
│    - Upper: trained_models/best_transformer_obstacle_model.pt   │
│    - Lower: ~/mpd-build/data_trained_models/...                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 3: Generate paper figures (optional)                      │
│  ─────────────────────────────────                              │
│  python scripts/visualize_transformer_denoising.py              │
│       ↓                                                         │
│  Output: results/transformer_denoising/*.png                    │
│  Note: Auto-loads pretrained model, no retraining needed        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Commands

```bash
# 1. Setup environment
cd ~/mpd-build && source set_env_variables.sh
conda activate mpd-splines-public
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd "/path/to/mpd-splines-public/dynamic_mpd"

# 2. Train upper model (first run, ~10 minutes)
python scripts/train_upper_diffusion_model.py --num_epochs 300 --num_train 10000

# 3. Run simulation
python scripts/rosorin_lidar_navigation.py

# 4. Generate paper figures (auto-loads model, no training)
python scripts/visualize_transformer_denoising.py

# 5. Run Isaac Sim 3D simulation (requires Isaac Sim installation)
cd "/home/wujiahao/ROSORIN_CAR and Reasearch/Diffusion-Guided Path Optimization (DGPO)"
./isaaclab_runner.sh "/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public/dynamic_mpd/scripts/rosorin_demo/isaac_sim_navigation.py" 2>&1
```

---

## 🤖 ROSOrin Robot Demo

This project supports deployment and validation of the dual-layer diffusion model on the ROSOrin mecanum wheel robot.

### ROSOrin Robot Parameters


| Parameter    | Value                         |
| ------------ | ----------------------------- |
| Robot Type   | Mecanum wheel omnidirectional |
| Chassis Size | Wheelbase 206mm               |
| LiDAR        | MS200 (360°, 12m range)      |
| Depth Camera | Aurora930                     |
| Controller   | Jetson Orin                   |
| Max Speed    | 0.5 m/s                       |

### Demo Script Description

#### 1. Pure Python Simulation (`scripts/rosorin_lidar_navigation.py`)

Pure Python simulation with matplotlib visualization, no Isaac Sim required.

**Features:**

- Real-time LiDAR simulation (180 rays, 12m range)
- Dynamic obstacles (circular motion, linear motion)
- Transformer upper diffusion model predicts obstacle trajectories
- Trajectory planning fusing potential field + diffusion prediction
- Fog of war map exploration visualization

**Run:**

```bash
cd ~/mpd-build && source set_env_variables.sh
conda activate mpd-splines-public
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/mpd-build/dynamic_mpd
python scripts/rosorin_lidar_navigation.py
```

#### 2. Isaac Sim 3D Simulation (`scripts/rosorin_demo/isaac_sim_navigation.py`)

Full 3D simulation running in NVIDIA Isaac Sim, requires Isaac Sim and Isaac Lab installation.

**Features:**

- 3D physics simulation environment (with environment lighting)
- USD scene creation (ground, walls, dynamic cylinder obstacles)
- Robot navigation task from one side to the other
- Real-time dynamic obstacle updates
- Transformer diffusion model integration
- Automatic visualization frame saving

**Run:**

```bash
# Use DGPO project's isaaclab_runner.sh to launch
cd "/home/wujiahao/ROSORIN_CAR and Reasearch/Diffusion-Guided Path Optimization (DGPO)"
./isaaclab_runner.sh "/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public/dynamic_mpd/scripts/rosorin_demo/isaac_sim_navigation.py"
```

**Optional Arguments:**

```bash
--headless  # Run in headless mode (no GUI, suitable for servers)
--num_envs N  # Number of parallel environments
```

### Result Storage


| Simulation Type | Output Directory            |
| --------------- | --------------------------- |
| Pure Python     | `results/lidar_navigation/` |
| Isaac Sim       | `results/isaac_sim_demo/`   |

---

## Project Structure

```
dynamic_mpd/
├── src/                                  # Core modules
│   ├── transformer_obstacle_diffusion.py # ⭐ Upper diffusion (Transformer, main)
│   ├── trainable_obstacle_diffusion.py   # Upper diffusion (MLP, baseline)
│   ├── complex_obstacle_data.py          # Training data generation (11 patterns)
│   └── mpd_integration.py                # Lower MPD model integration
│
├── scripts/                              # Scripts
│   ├── train_upper_diffusion_model.py    # ⭐ [Step 1] Train upper model
│   ├── rosorin_lidar_navigation.py       # ⭐ [Step 2] LiDAR navigation sim
│   ├── visualize_transformer_denoising.py# ⭐ [Step 3] Paper figure generation
│   ├── compare_transformer_mlp.py        # (Optional) Transformer vs MLP
│   ├── train_full_comparison.py          # (Optional) Full comparison exp
│   ├── visualize_complex_training.py     # (Archived) MLP version viz
│   └── rosorin_demo/                     # ⭐ ROSOrin robot demos
│       ├── isaac_sim_navigation.py       # Isaac Sim 3D navigation
│       └── rosorin_lidar_navigation.py   # Pure Python 2D navigation
│
├── trained_models/                       # ⭐ Trained model weights
│   ├── best_transformer_obstacle_model.pt   # Best upper model
│   └── final_transformer_obstacle_model.pt  # Final upper model
│
├── results/                              # Output results
│   ├── transformer_denoising/            # Paper figure output
│   ├── lidar_navigation/                 # Simulation results
│   └── isaac_sim_demo/                   # Isaac Sim demo frames
│
└── README.md
```

---

## ⚠️ Important: mpd-build & Pretrained Models

### Project Directory Relationship

This project depends on the original MPD paper's pretrained models. There are **two related directories**:


| Directory              | Path           | Description                                       |
| ---------------------- | -------------- | ------------------------------------------------- |
| **mpd-build**          | `~/mpd-build/` | Original paper's build directory with pretraining |
| **mpd-splines-public** | Parent dir     | Our dynamic obstacle avoidance extension          |

### Pretrained Model Locations

**Lower Diffusion Model (Trajectory Planning) - Original MPD Pretrained:**

```bash
# Symbolic link structure
~/mpd-build/data_trained_models -> ~/mpd-build/data_public/data_trained_models

# Actual model path
~/mpd-build/data_public/data_trained_models/
├── launch_train_diffusion_models-v04_2024-09-17_21-44-17/
│   └── generative_model_class___GaussianDiffusionModel/
│       ├── dataset_subdir___EnvSimple2D-RobotPointMass2D.../
│       │   └── .../checkpoints/
│       │       ├── model_current.pth          # Lower diffusion model
│       │       └── ema_model_current.pth      # EMA version (recommended)
│       ├── dataset_subdir___EnvDense2D-RobotPointMass2D.../
│       └── dataset_subdir___EnvNarrowPassageDense2D.../
└── ...
```

**Upper Diffusion Model (Obstacle Prediction) - Our Training:**

```bash
# Our trained obstacle prediction model
dynamic_mpd/trained_models/best_transformer_obstacle_model.pt
```

### Configuration File Path

Config files in scripts are located at:

```bash
~/mpd-build/scripts/inference/cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml
```

Config files use `${HOME}/mpd-build/...` as model path prefix.

### Environment Setup Before Running

```bash
# 1. Enter mpd-build and set environment variables
cd ~/mpd-build
source set_env_variables.sh

# 2. Activate conda environment
conda activate mpd-splines-public

# 3. Set dynamic library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. Run scripts
cd "/path/to/mpd-splines-public/dynamic_mpd"
python scripts/rosorin_lidar_navigation.py
```

### Model Usage Summary


| Model Layer         | Function         | Architecture | Source              | Path                                                        |
| ------------------- | ---------------- | ------------ | ------------------- | ----------------------------------------------------------- |
| **Upper**           | Obstacle predict | Transformer  | Our training        | `trained_models/best_transformer_obstacle_model.pt`         |
| **Upper(baseline)** | Obstacle predict | MLP          | Our training        | `old_results/trained_diffusion_model.pth`                   |
| **Lower**           | Robot trajectory | UNet         | Original pretrained | `~/mpd-build/data_trained_models/.../ema_model_current.pth` |

### Quick Start

#### Method 1: 2D LiDAR Simulation (Pure Python, recommended for quick test)

```bash
# 1. Setup environment
cd ~/mpd-build && source set_env_variables.sh
conda activate mpd-splines-public
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 2. Enter project directory
cd ~/mpd-build/dynamic_mpd

# 3. Train upper model (first run, ~30 minutes)
python scripts/train_upper_diffusion_model.py --num_epochs 300 --num_train 10000

# 4. Run 2D simulation (with LiDAR visualization)
python scripts/rosorin_lidar_navigation.py

# 5. Generate paper figures
python scripts/visualize_transformer_denoising.py
```

#### Method 2: Isaac Sim 3D Simulation (requires Isaac Lab installation)

```bash
# 1. Enter DGPO project directory (contains isaaclab_runner.sh)
cd "/home/wujiahao/ROSORIN_CAR and Reasearch/Diffusion-Guided Path Optimization (DGPO)"

# 2. Run Isaac Sim navigation demo
./isaaclab_runner.sh "/home/wujiahao/ROSORIN_CAR and Reasearch/Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models/mpd-splines-public/dynamic_mpd/scripts/rosorin_demo/isaac_sim_navigation.py"

# Optional arguments:
#   --headless      Run in headless mode
#   --max_steps 800 Set maximum simulation steps
```

**Isaac Sim Scene Description:**

- Arena: 4m × 4m plane with environment and directional lighting
- Robot: ROSOrin mecanum wheel robot (simplified as blue cube)
- Obstacles: 5 dynamic moving cylinders with different motion patterns:
  - `circular`: Circular motion
  - `linear_x/y`: X/Y axis reciprocating motion
  - `zigzag`: Zigzag motion
  - `oscillate`: Oscillation motion
- Task: Navigate from bottom-left to top-right, avoiding all dynamic obstacles

### Load Pretrained Model (Python API)

```python
from src.transformer_obstacle_diffusion import TransformerObstacleDiffusionWrapper

# Method 1: Load from checkpoint
model = TransformerObstacleDiffusionWrapper.load_from_checkpoint(
    'trained_models/best_transformer_obstacle_model.pt',
    device='cuda'
)

# Method 2: Create first then load
model = TransformerObstacleDiffusionWrapper(device='cuda')
model.load('trained_models/best_transformer_obstacle_model.pt')

# Predict obstacle future trajectory
# history: [8, 2] past 8 frames positions
mean_pred, std_pred, samples = model.predict(history, num_samples=20)
# mean_pred: [12, 2] future 12 steps prediction mean
```

---

### Pretrained Model Location (Local Copy)

If `~/mpd-build` is unavailable, use the copy in current project:

```
mpd-splines-public/data_public/data_trained_models/
├── launch_train_diffusion_models-v04_2024-09-17_21-44-17/
│   └── generative_model_class___GaussianDiffusionModel/
│       ├── dataset_subdir___EnvSimple2D-RobotPointMass2D.../
│       │   └── checkpoints/
│       │       ├── model_current.pth          # Lower diffusion model
│       │       └── ema_model_current.pth      # EMA version
│       ├── dataset_subdir___EnvDense2D-RobotPointMass2D.../
│       ├── dataset_subdir___EnvNarrowPassageDense2D.../
│       └── dataset_subdir___EnvPlanar2Link.../
└── ...
```


| Environment             | Usage       | Model Path                                                         |
| ----------------------- | ----------- | ------------------------------------------------------------------ |
| EnvSimple2D             | Simple 2D   | `.../EnvSimple2D.../checkpoints/ema_model_current.pth`             |
| EnvDense2D              | Dense obs   | `.../EnvDense2D.../checkpoints/ema_model_current.pth`              |
| EnvNarrowPassageDense2D | Narrow pass | `.../EnvNarrowPassageDense2D.../checkpoints/ema_model_current.pth` |

---

## 📊 Experimental Results (For Paper)

### Experiment Configuration


| Configuration       | Value                                          |
| ------------------- | ---------------------------------------------- |
| Training Samples    | 10,000                                         |
| Test Samples        | 2,000                                          |
| Training Epochs     | 200                                            |
| Batch Size          | 64                                             |
| Learning Rate       | 1e-4                                           |
| Observation History | 8 frames                                       |
| Prediction Horizon  | 12 steps                                       |
| State Dimension     | 2 (x, y)                                       |
| Diffusion Steps     | 50                                             |
| Motion Types        | circular, linear, zigzag, spiral, acceleration |
| GPU                 | NVIDIA RTX (CUDA)                              |
| Random Seed         | 42                                             |

### Main Results Comparison


| Model                  | Final ADE ↓   | Final FDE ↓   | Best ADE ↓    | Training Time |
| ---------------------- | -------------- | -------------- | -------------- | ------------- |
| MLP Baseline           | 1.1250         | 1.1300         | 0.6346         | ~12 min       |
| **Transformer (Ours)** | **0.0726**     | **0.1534**     | **0.0661**     | ~11 min       |
| **Improvement**        | **93.5%** ⬆️ | **86.4%** ⬆️ | **89.6%** ⬆️ | -             |

> **ADE** (Average Displacement Error): Average Euclidean distance between all predicted points and ground truth
> **FDE** (Final Displacement Error): Euclidean distance between predicted endpoint and ground truth endpoint

### Training Convergence Curves

```
Training Loss Evolution:
                                                                
  MLP Baseline:     ████████████████████████████████ 1.0057 (epoch 200)
  Transformer:      █ 0.0384 (epoch 200)
              
  Conclusion: Transformer converges to lower loss with same epochs
```

### Performance by Motion Type


| Motion Type  | MLP ADE | Transformer ADE | Improvement |
| ------------ | ------- | --------------- | ----------- |
| Circular     | ~1.2    | ~0.08           | 93.3%       |
| Linear       | ~0.9    | ~0.05           | 94.4%       |
| Zigzag       | ~1.3    | ~0.09           | 93.1%       |
| Spiral       | ~1.1    | ~0.07           | 93.6%       |
| Acceleration | ~1.0    | ~0.08           | 92.0%       |

### Model Parameter Comparison


| Parameter       | MLP Baseline | Transformer     |
| --------------- | ------------ | --------------- |
| Total Params    | 1,462,808    | 1,788,546       |
| Param Increase  | -            | +22.3%          |
| Hidden Dim      | 256          | 128 (d_model)   |
| Network Layers  | 6 (FC)       | 4 (Transformer) |
| Attention Heads | -            | 4               |
| FFN Dim         | -            | 512             |
| Dropout         | 0.1          | 0.1             |

### Transformer Architecture Advantages


| Component                        | Function                               | Performance Contribution              |
| -------------------------------- | -------------------------------------- | ------------------------------------- |
| **Temporal Self-Attention**      | Capture temporal dependencies in hist  | Understand motion trends & accel      |
| **Adaptive LayerNorm**           | Inject diffusion timestep into network | More effective conditional denoising  |
| **Sinusoidal Position Encoding** | Encode sequence position information   | Generalize to variable-length inputs  |
| **Cross-Attention** (optional)   | Model interactions between obstacles   | Handle complex multi-target scenarios |

### Visualization Results

Experiment results saved in `results/full_comparison/`:


| File                            | Content                            |
| ------------------------------- | ---------------------------------- |
| `training_curves.png`           | MLP vs Transformer training curves |
| `prediction_by_motion_type.png` | Prediction visualization by motion |
| `results_table.png`             | Experiment results summary table   |

### Reproduction Commands

```bash
# Activate environment
cd ~/mpd-build && source set_env_variables.sh
conda activate mpd-splines-public
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run full comparison experiment
cd "/path/to/mpd-splines-public/dynamic_mpd"
python scripts/train_full_comparison.py
```

### Paper Table Format (LaTeX)

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

## 🧠 Transformer Diffusion Model Innovations

### Architecture Design (`src/transformer_obstacle_diffusion.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│               Transformer Obstacle Diffusion                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Temporal Transformer Encoder                             │  │
│  │  • Input Embedding + Sinusoidal Position Encoding         │  │
│  │  • Self-Attention: Capture temporal dependencies          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Cross-Attention: Model multi-obstacle interactions       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Diffusion Transformer Decoder                            │  │
│  │  • Adaptive LayerNorm: Inject diffusion timestep info     │  │
│  │  • Output: pred_horizon × state_dim                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Innovations


| Innovation                  | Technical Details                 | Advantage over MLP             |
| --------------------------- | --------------------------------- | ------------------------------ |
| **Temporal Self-Attention** | Multi-head attention for temporal | Not limited by fixed window    |
| **Cross-Attention**         | Model obstacle interactions       | Dynamically perceive scene     |
| **Adaptive LayerNorm**      | Timestep-modulated normalization  | More effective than concat     |
| **Sinusoidal Position**     | Continuous position encoding      | Generalize to variable lengths |

---

## Paper Contribution Summary

1. **Dual-Layer Diffusion Architecture**: Upper predicts obstacle motion, lower plans collision-free trajectories
2. **Transformer Obstacle Prediction**: 93.5% improvement over MLP (ADE)
3. **Repulsive Field Guidance**: Inject physical constraints during diffusion process
4. **Hybrid Scoring Mechanism**: Multi-objective heuristic trajectory selection
5. **Unknown Environment Exploration**: LiDAR perception + fog of war map

### Experimental Baselines


| Baseline Method    | Description                  | Comparison Dimension    |
| ------------------ | ---------------------------- | ----------------------- |
| MLP Diffusion      | Fully-connected denoiser     | Prediction accuracy     |
| Linear Predictor   | Linear extrapolation         | Dynamic obstacle        |
| LSTM/GRU           | Recurrent neural networks    | Temporal modeling       |
| Social Force Model | Social force model           | Multi-obstacle interact |
| MPD (original)     | Static environment diffusion | Dynamic adaptability    |

### Key Citations

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

## 🔄 Changelog


| Date       | Version | Updates                                              |
| ---------- | ------- | ---------------------------------------------------- |
| 2026-01-29 | v0.3.0  | Add Transformer upper diffusion, complete comparison |
| 2026-01-28 | v0.2.5  | Optimize near-goal avoidance strategy                |
| 2026-01-27 | v0.2.0  | Reorganize code structure, archive old scripts       |
| 2026-01-22 | v0.1.0  | Initial dual-layer diffusion architecture            |

---

## 🆕 Visualization Effects

### Fog of War Map


| Region              | Color                      | Description                           |
| ------------------- | -------------------------- | ------------------------------------- |
| Unexplored          | Dark gray semi-transparent | Unscanned by robot                    |
| Explored            | Light green gradient       | Safe regions scanned by LiDAR         |
| Obstacle (detected) | Bright color               | Current obstacles in LiDAR range      |
| Obstacle (history)  | Faded color                | Previously detected, now out of range |

### LiDAR Ray Colors


| Color | Meaning                          |
| ----- | -------------------------------- |
| Cyan  | Ray not hitting obstacle (free)  |
| Red   | Ray hitting obstacle (detection) |

### Diffusion Convergence Visualization


| Color            | Meaning                     |
| ---------------- | --------------------------- |
| Red trajectory   | Initial noise state (t=100) |
| Blue trajectory  | Converged state (t=0)       |
| Green trajectory | Best selected trajectory    |

---

## Notes

1. **Pretrained Models**: Lower diffusion model uses original paper pretrained weights (in `data_trained_models/`)
2. **GPU Memory**: GPU training recommends ≥4GB
3. **Random Seed**: All scripts use `SEED=42` for reproducibility
4. **Dependencies**: `pip install torch numpy matplotlib scipy pillow`

---

## 🙏 Acknowledgments

This project builds upon the excellent work of the **Motion Planning Diffusion (MPD)** framework. We gratefully acknowledge the original authors for their pioneering contributions and for making their code and pretrained models publicly available.

### Original MPD Project

- **Repository**: [https://github.com/joaoamcarvalho/mpd-splines-public](https://github.com/joaoamcarvalho/mpd-splines-public)
- **Paper**: *Motion Planning Diffusion: Learning and Adapting Robot Motion Planning with Diffusion Models*
- **Authors**: João Carvalho, An T. Le, Piotr Kicki, Dorothea Koert, Jan Peters
- **Published in**: IEEE Transactions on Robotics, 2025

Our project extends the original MPD framework by introducing a **dual-layer diffusion architecture** for dynamic obstacle avoidance. Specifically, we utilize the **pretrained GaussianDiffusion trajectory planning model** from the original MPD project as our **lower-layer diffusion model** for robot trajectory generation, and build a novel **Transformer-based upper-layer diffusion model** on top of it for dynamic obstacle trajectory prediction. The original pretrained model weights (located in `data_trained_models/`) are essential for the lower-layer trajectory planning component of our system.

We sincerely thank the MPD authors for their outstanding contribution to the robotics and diffusion model community.

```bibtex
@article{carvalho2025motion,
  title={Motion planning diffusion: Learning and adapting robot motion planning with diffusion models},
  author={Carvalho, Jo{\~a}o and Le, An T and Kicki, Piotr and Koert, Dorothea and Peters, Jan},
  journal={IEEE Transactions on Robotics},
  year={2025},
  publisher={IEEE}
}
```

---

## License

MIT License

## Author

Dynamic MPD Project

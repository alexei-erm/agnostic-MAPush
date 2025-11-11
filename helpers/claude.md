# Claude Context: agnostic-MAPush Repository

**Quick Start for Claude Sessions** - Read this first to understand the codebase structure and workflow.

---

## Project Overview

**Purpose**: Hierarchical Multi-Agent Reinforcement Learning (MARL) framework for training multiple quadrupedal robots (Unitree Go1) to collaboratively push objects toward targets in Isaac Gym physics simulation.

**Paper**: "Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing" (arXiv 2411.07104)
**Website**: https://collaborative-mapush.github.io/

**Key Concept**: 2 Go1 robots learn to coordinate pushing different shaped objects (cuboid, cylinder, T-block) to target locations using MAPPO (Multi-Agent PPO).

---

## Directory Structure

```
agnostic-MAPush/
├── mqe/                          # Core multi-agent quadruped environment framework
│   ├── envs/
│   │   ├── base/                 # Base task and robot classes
│   │   ├── go1/                  # Go1 robot implementation (main robot)
│   │   ├── wrappers/             # Observation/action/reward wrappers
│   │   └── configs/              # Task configs (mid/upper level controllers)
│   └── utils/                    # Terrain, math, observation utils
├── openrl_ws/                    # Training and testing workspace
│   ├── train.py                  # Main training script
│   ├── test.py                   # Evaluation script (success rate, videos)
│   ├── update_config.py          # Copies task config to mqe/envs/configs/
│   └── cfgs/                     # Algorithm configs (ppo.yaml, mappo, mat)
├── task/                         # Task-specific configurations
│   ├── cuboid/                   # Small box pushing (0.6x0.6m)
│   │   ├── config.py             # Task parameters, rewards, domain randomization
│   │   └── train.sh              # Training launcher script
│   ├── Tblock/                   # T-shaped block pushing
│   └── cylinder/                 # Cylinder pushing
├── resources/                    # Assets and pretrained models
│   ├── robots/                   # Robot URDFs (Go1, A1, ANYmal, Cassie)
│   ├── objects/                  # Object URDFs (boxes, cylinders, targets)
│   ├── command_nets/             # Pretrained locomotion policies
│   └── actuator_nets/            # Low-level motor control (unitree_go1.pt)
├── results/                      # Training outputs (checkpoints, logs, metrics)
│   └── MM-DD-HH_<task>/          # Timestamped result directories
├── helpers/                      # Utility scripts
│   ├── gpu_monitor.py            # GPU memory monitoring
│   ├── architecture_check.py     # Model inspection tool
│   └── claude.md                 # This file
└── docs/                         # Documentation and videos
```

---

## Key Entry Points

### Training Pipeline

**Start training a task:**
```bash
source task/cuboid/train.sh False
```

**What happens:**
1. `update_config.py` copies `task/cuboid/config.py` → `mqe/envs/configs/go1_push_mid_config.py`
2. `train.py` creates Isaac Gym with 500 parallel environments (2 agents each)
3. Trains MAPPO for 180M timesteps, saves checkpoints every 20M steps
4. Evaluates each checkpoint and outputs success rate to `results/<timestamp>_cuboid/success_rate.txt`

**Main training script:** `openrl_ws/train.py`
- Uses OpenRL PPOAgent/PPONet
- Logs to `./log/MQE/go1push_mid/<task>/run<N>/`
- Saves results to `./results/<timestamp>_<task>/`

### Testing/Evaluation

**Test a trained model:**
```bash
source results/<timestamp>_cuboid/task/train.sh True
```

**Evaluation script:** `openrl_ws/test.py`
- Two modes:
  - `calculator`: Compute success rate over 300 episodes
  - `viewer`: Visual rendering with optional video recording
- Metrics: success_rate, finish_time, collision_degree, collaboration_degree

---

## Configuration System

### Task Configuration Hierarchy

1. **Task-specific config**: `task/<task_name>/config.py`
   - Inherits from `Go1PushMidCfg`
   - Defines: object URDF, reward scales, domain randomization, termination conditions

2. **Mid-level config**: `mqe/envs/configs/go1_push_mid_config.py` (auto-generated)
   - 2 agents, 2 NPCs (object + target), 20s episodes
   - Used for collaborative pushing controller

3. **Upper-level config**: `mqe/envs/configs/go1_push_upper_config.py`
   - Single agent (2 robots as one unit), 160s episodes
   - High-level goal planning with obstacles and multi-stage targets

### Algorithm Configurations

Located in `openrl_ws/cfgs/`:
- `ppo.yaml`: Standard PPO (lr=5e-3)
- `dppo.yaml`: Distributed PPO
- `mat.yaml`: Multi-Agent Transformer (lr=7e-4)
- `jrpo.yaml`: Joint reward/policy optimization

---

## Environment Architecture

### Wrapper Chain (bottom to top)
```
Go1Object (physics simulation, loads object URDF)
    ↓
LeggedRobotField (terrain, rendering)
    ↓
Go1 (robot control, locomotion policy)
    ↓
EmptyWrapper (agent/NPC info)
    ↓
Go1PushMidWrapper (observations, actions, rewards)
    ↓
mqe_openrl_wrapper (OpenRL compatibility)
    ↓
PPOAgent (RL algorithm)
```

### Observation/Action Spaces

**Mid-Level Controller (per agent):**
- **Observation**: `(5,)` = [2D relative goal distance, 3D positions of 2 agents]
- **Action**: `(3,)` = [vx, vy, angular_vel] velocity commands
- **Reward Components** (8 terms):
  - Target distance reward
  - Approach reward
  - Collision punishment (inter-robot)
  - Push reward
  - Optimal Collaborative Behavior (OCB) reward
  - Exception punishment (out of bounds, etc.)
  - Reach target bonus

**High-Level Controller (single "unit"):**
- **Observation**: `(26,)` = object/target/robot positions
- **Action**: `(2,)` = 2D goal position commands

---

## Available Tasks

### 1. Cuboid (`task/cuboid/`)
- 2 agents push small box (0.6x0.6m) to target
- Default task for most experiments

### 2. T-block (`task/Tblock/`)
- T-shaped object with asymmetric geometry
- Requires coordinated pushing strategy

### 3. Cylinder (`task/cylinder/`)
- Cylindrical object (prone to rolling)
- More challenging dynamics

**Each task has:**
- `config.py`: Task-specific parameters
- `train.sh`: Training launcher script

---

## Important Files to Know

### Training & Evaluation
- `openrl_ws/train.py:make_env()` - Creates Isaac Gym environment
- `openrl_ws/test.py:calculate()` - Computes success rate metrics
- `openrl_ws/utils.py:parse_config_args()` - Argument parsing

### Environment Core
- `mqe/envs/wrappers/Go1PushMidWrapper.py` - Main task wrapper (obs/action/reward)
- `mqe/envs/go1/go1_object.py` - Object loading and management
- `mqe/envs/go1/go1_push.py` - Go1 pushing environment

### Configuration
- `task/cuboid/config.py` - Example task config (extend `Go1PushMidCfg`)
- `mqe/envs/configs/go1_push_mid_config.py` - Auto-generated mid-level config

### Utilities
- `mqe/utils/helpers.py` - Config merging, class conversions
- `mqe/utils/task_registry.py` - Task registration system
- `helpers/gpu_monitor.py` - GPU memory tracking

---

## Common Workflows

### Training a New Task
1. Create `task/<new_task>/config.py` extending `Go1PushMidCfg`
2. Set object URDF, reward scales, domain randomization
3. Create `task/<new_task>/train.sh` (copy from cuboid and modify)
4. Run: `source task/<new_task>/train.sh False`
5. Monitor: `tensorboard --logdir=./log/`

### Evaluating a Checkpoint
1. Find checkpoint in `results/<timestamp>_<task>/rl_model_<steps>_steps/`
2. Run: `source results/<timestamp>_<task>/task/train.sh True`
3. Check success rate in `results/<timestamp>_<task>/success_rate.txt`

### Modifying Rewards
1. Edit `task/<task>/config.py` under `class rewards:`
2. Update scale values (e.g., `target_dist_reward_scale = 100.0`)
3. Re-run training

### Debugging Training
1. Check GPU usage: `python helpers/gpu_monitor.py`
2. Inspect model architecture: `python helpers/architecture_check.py <checkpoint_path>`
3. View TensorBoard: `tensorboard --logdir=./log/`

---

## Key Metrics

**Success Rate**: Percentage of episodes where object reaches target within episode length

**Finish Time**: Average time (seconds) to complete successful episodes

**Collision Degree**: Average inter-robot collision frequency per timestep

**Collaboration Degree**: Measure of coordinated pushing behavior (from OCB reward)

---

## Technologies

**Core Dependencies:**
- Isaac Gym Preview 4 (NVIDIA physics simulator)
- PyTorch (CUDA-enabled)
- OpenRL (MARL framework)
- Gymnasium/PettingZoo (RL APIs)

**RL Algorithms:**
- MAPPO (Multi-Agent PPO) - primary
- PPO, DPPO, MAT, JRPO - alternatives

---

## Quick Reference: File Paths

```python
# Main scripts
TRAIN_SCRIPT = "openrl_ws/train.py"
TEST_SCRIPT = "openrl_ws/test.py"
UPDATE_CONFIG = "openrl_ws/update_config.py"

# Task configs
TASK_CONFIGS = "task/<task_name>/config.py"
GENERATED_CONFIG = "mqe/envs/configs/go1_push_mid_config.py"

# Wrappers
MAIN_WRAPPER = "mqe/envs/wrappers/Go1PushMidWrapper.py"

# Resources
ROBOT_URDF = "resources/robots/go1/urdf/go1.urdf"
ACTUATOR_NET = "resources/actuator_nets/unitree_go1.pt"
OBJECT_URDFS = "resources/objects/*.urdf"

# Results
RESULTS_DIR = "results/<MM-DD-HH>_<task>/"
CHECKPOINTS = "results/<timestamp>_<task>/rl_model_<steps>_steps/module.pt"
SUCCESS_RATES = "results/<timestamp>_<task>/success_rate.txt"
```

---

## Git Status Notes

Current branch: `main`

**Modified files:**
- `results/11-07-17_cuboid/success_rate.txt` - Latest evaluation results
- `task/cuboid/train.sh` - Training script modifications

**Untracked:**
- `VNC_SETUP_STATUS.md` - VNC setup documentation

Recent work focused on cuboid task training and evaluation.

---

## Tips for Future Claude Sessions

1. **Always check the task config first** when debugging training issues
2. **Results are timestamped** - look for latest directory in `results/`
3. **Training is GPU-intensive** - expect 500 parallel Isaac Gym environments
4. **Checkpoints are large** (~100MB each) - saved every 20M steps
5. **Success rate files** contain evaluation history across all checkpoints
6. **Use gpu_monitor.py** to debug OOM errors
7. **Don't modify `mqe/envs/configs/go1_push_mid_config.py` directly** - it's auto-generated

---

**Last Updated**: 2025-11-11
**Repository**: /home/gvlab/agnostic-MAPush

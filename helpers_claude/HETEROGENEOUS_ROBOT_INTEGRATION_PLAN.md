# Heterogeneous Robot Integration Plan
# Go1 + Different Robot for Collaborative Pushing

**Date Created**: 2025-11-15
**Status**: 📋 Planning Phase
**Goal**: Enable HAPPO to train with different robot types (e.g., Go1 + manipulator, Go1 + wheeled robot)

---

## Table of Contents
1. [Overview](#overview)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [Key Challenges](#key-challenges)
4. [Implementation Approach](#implementation-approach)
5. [Code Modifications Required](#code-modifications-required)
6. [Testing Strategy](#testing-strategy)
7. [Timeline Estimate](#timeline-estimate)

---

## Overview

### Objective
After obtaining satisfactory HAPPO baseline performance with 2 × Go1 quadrupeds pushing a cuboid, extend the system to support:
- **1 × Go1 quadruped** (existing)
- **1 × Different robot** (new - could be manipulator, wheeled robot, bipedal, etc.)

### Why HAPPO is Perfect for This
HAPPO (Heterogeneous-Agent PPO) is **specifically designed** for heterogeneous agents:
- ✅ **Different observation spaces** per agent
- ✅ **Different action spaces** per agent
- ✅ **Different network architectures** per agent
- ✅ **Sequential updates** that handle agent asymmetry
- ✅ **No parameter sharing requirement** (unlike MAPPO)

**This is literally what HAPPO was made for!** 🎯

---

## Current Architecture Analysis

### 1. Current Homogeneous Setup (2 × Go1)

```
Environment: Go1Object (Isaac Gym)
├── Agent 0: Go1 quadruped
│   ├── Observation: (8,) - [target_dist, target_angle, box_relative_pos, other_agent_pos]
│   ├── Action: (3,) - [vx, vy, vyaw] velocity commands
│   └── DOF: 12 joint positions
└── Agent 1: Go1 quadruped (identical)
    ├── Observation: (8,) - same structure
    ├── Action: (3,) - same structure
    └── DOF: 12 joint positions (same)
```

**Key Assumptions in Current Code:**
- All agents have **same number of DOFs** (12)
- All agents have **same observation space** (8,)
- All agents have **same action space** (3,)
- All agents use **same locomotion policy** (Go1 actuator network)
- All agents have **same body structure** (4 legs, same link names)

### 2. Proposed Heterogeneous Setup (Go1 + Robot X)

```
Environment: HeterogeneousPushEnv (Isaac Gym)
├── Agent 0: Go1 quadruped
│   ├── Observation: (8,) - [target_dist, target_angle, box_relative_pos, other_agent_pos]
│   ├── Action: (3,) - [vx, vy, vyaw] velocity commands
│   └── DOF: 12 joint positions
└── Agent 1: Robot X (e.g., 2-arm manipulator)
    ├── Observation: (N,) - DIFFERENT! [target_dist, box_pos, ee_pos, joint_states, ...]
    ├── Action: (M,) - DIFFERENT! Could be [joint_velocities] or [end_effector_pose]
    └── DOF: K (e.g., 6 per arm = 12, but could be different)
```

---

## Key Challenges

### Challenge 1: Different Action Spaces ⚠️

**Current Code Assumes:**
```python
# mqe/envs/base/legged_robot.py
self.num_actions = self.num_agents * cfg.env.num_actions  # Assumes all agents same!
```

**Problem:**
- Go1 action: (3,) velocity commands
- Manipulator action: Could be (12,) joint positions, or (7,) end-effector pose
- Current code uses `num_agents * num_actions` which assumes uniformity

**Solution Needed:**
- Per-agent action space definition
- `num_actions` → `num_actions_per_agent = [3, 12]`

### Challenge 2: Different Observation Spaces ⚠️

**Current Code:**
```python
# mqe/envs/wrappers/go1_push_mid_wrapper.py
# Assumes all agents get same observation structure
obs_dim = 2 + 3 * num_agents  # Same for all
```

**Problem:**
- Go1 observation: Velocity commands + body state
- Manipulator observation: Joint states + end-effector poses
- Different sensors, different proprioception

**Solution Needed:**
- Per-agent observation space definition
- Agent-specific observation buffers

### Challenge 3: Different Robot Assets (URDF/MJCF) ⚠️

**Current Code:**
```python
# mqe/envs/base/legged_robot.py
# Loads same asset for all agents
self.robot_asset = self.gym.load_asset(...)
for i in range(self.num_envs):
    for j in range(self.num_agents):
        self.gym.create_actor(..., self.robot_asset, ...)  # Same asset!
```

**Problem:**
- Different robots = different URDF files
- Different link names, joint names, collision shapes
- Different mass properties, inertias

**Solution Needed:**
- Load multiple assets: `robot_assets = [go1_asset, manipulator_asset]`
- Agent-specific asset loading in environment creation

### Challenge 4: Different Low-Level Controllers ⚠️

**Current Code:**
```python
# mqe/envs/go1/go1.py
self._prepare_locomotion_policy()  # Loads Go1 actuator network
# Applied to ALL agents
```

**Problem:**
- Go1 uses pretrained locomotion policy (walk-these-ways)
- Manipulator might use:
  - IK solver for end-effector control
  - Direct joint control
  - Different pretrained policy
- Can't share same controller

**Solution Needed:**
- Per-agent controller architecture
- Agent-specific policy loading

### Challenge 5: Different DOFs per Agent ⚠️

**Current Code:**
```python
# mqe/envs/base/legged_robot.py
self.num_dof = 12  # Hardcoded for Go1
self.dof_pos = torch.zeros(self.num_envs * self.num_agents, self.num_dof, ...)
```

**Problem:**
- Go1: 12 DOFs (3 per leg × 4 legs)
- Manipulator: Could be 6, 7, 12, 14, etc.
- Current code assumes `num_agents × same_dof`

**Solution Needed:**
- Variable DOF per agent: `dof_per_agent = [12, 14]`
- Ragged tensor handling or agent-specific buffers

---

## Implementation Approach

### Phase 1: Architecture Redesign (Core Infrastructure)

#### 1.1 Create Heterogeneous Base Class

**New File**: `mqe/envs/base/heterogeneous_robot.py`

```python
class HeterogeneousRobot(LeggedRobot):
    """Base class for heterogeneous multi-robot environments."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Per-agent specifications
        self.agent_types = cfg.env.agent_types  # e.g., ["go1", "manipulator"]
        self.num_agents_per_type = cfg.env.num_agents_per_type  # e.g., [1, 1]

        # Per-agent action/observation spaces
        self.action_spaces_per_agent = []
        self.observation_spaces_per_agent = []

        # Per-agent assets
        self.robot_assets = []

        # Per-agent controllers
        self.controllers = []

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _load_robot_assets(self):
        """Load different URDF/MJCF for each agent type."""
        for agent_type in self.agent_types:
            asset_path = self.cfg.asset.files[agent_type]
            asset = self.gym.load_asset(self.sim, asset_path, ...)
            self.robot_assets.append(asset)

    def _create_envs(self):
        """Create environments with heterogeneous agents."""
        for i in range(self.num_envs):
            env = self.gym.create_env(...)

            # Create each agent with its specific asset
            agent_idx = 0
            for agent_type_idx, agent_type in enumerate(self.agent_types):
                for j in range(self.num_agents_per_type[agent_type_idx]):
                    asset = self.robot_assets[agent_type_idx]
                    actor = self.gym.create_actor(env, asset, ...)
                    agent_idx += 1
```

#### 1.2 Agent-Specific Configuration

**New File**: `mqe/envs/configs/heterogeneous_push_config.py`

```python
class HeterogeneousPushCfg(Go1Cfg):

    class env(Go1Cfg.env):
        num_agents = 2
        agent_types = ["go1", "manipulator"]  # NEW!
        num_agents_per_type = [1, 1]  # NEW!

        # Per-agent specifications
        num_observations_per_agent = [8, 16]  # NEW!
        num_actions_per_agent = [3, 12]  # NEW!

    class asset:
        # Multiple asset files
        files = {
            "go1": "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf",
            "manipulator": "{LEGGED_GYM_ROOT_DIR}/resources/robots/dual_arm/urdf/dual_arm.urdf"
        }

        # Per-agent DOFs
        dof_per_agent = {
            "go1": 12,
            "manipulator": 14  # 7 per arm
        }

    class control:
        # Per-agent control types
        control_types = {
            "go1": "C",  # Command-based (velocity)
            "manipulator": "P"  # Position control
        }

        # Per-agent controllers
        controllers = {
            "go1": "locomotion_policy",
            "manipulator": "ik_solver"
        }
```

### Phase 2: HARL Integration

#### 2.1 Update MAPush Environment Wrapper

**File**: `HARL/harl/envs/mapush/mapush_env.py`

```python
class MAPushEnv:
    """HARL wrapper for heterogeneous MAPush environment."""

    def __init__(self, env_args):
        # ... existing code ...

        # Check if heterogeneous
        self.is_heterogeneous = env_args.get("heterogeneous", False)

        if self.is_heterogeneous:
            # Different observation spaces per agent
            self.observation_space = [
                spaces.Box(low=-np.inf, high=np.inf,
                          shape=(self.env.num_observations_per_agent[i],))
                for i in range(self.n_agents)
            ]

            # Different action spaces per agent
            self.action_space = [
                spaces.Box(low=-1.0, high=1.0,
                          shape=(self.env.num_actions_per_agent[i],))
                for i in range(self.n_agents)
            ]
        else:
            # Existing homogeneous code
            # ...
```

**Key Point**: HARL **already supports** list-based action/observation spaces!

#### 2.2 HAPPO Actor Networks

HAPPO will automatically create **separate actor networks** for each agent:

```python
# HARL/harl/algorithms/actors/happo.py
# Already supports this!
for agent_id in range(num_agents):
    self.actor[agent_id] = ActorNetwork(
        obs_space=env.observation_space[agent_id],  # Different!
        act_space=env.action_space[agent_id],       # Different!
        hidden_sizes=[256, 256]
    )
```

**No HARL code changes needed!** 🎉

### Phase 3: Environment-Specific Implementation

#### 3.1 Create Heterogeneous Pushing Environment

**New File**: `mqe/envs/heterogeneous/heterogeneous_push.py`

```python
from mqe.envs.base.heterogeneous_robot import HeterogeneousRobot

class HeterogeneousPush(HeterogeneousRobot):
    """Pushing task with Go1 + different robot."""

    def _init_agents(self):
        """Initialize each agent with its specific configuration."""

        # Agent 0: Go1
        self.go1_controller = self._load_go1_locomotion_policy()

        # Agent 1: Manipulator
        self.manipulator_controller = self._load_manipulator_controller()

    def compute_observations(self):
        """Compute per-agent observations."""
        obs_list = []

        # Agent 0 (Go1) observations
        go1_obs = self._compute_go1_obs()  # Shape: (num_envs, 8)
        obs_list.append(go1_obs)

        # Agent 1 (Manipulator) observations
        manip_obs = self._compute_manipulator_obs()  # Shape: (num_envs, 16)
        obs_list.append(manip_obs)

        # Stack along agent dimension
        return torch.stack(obs_list, dim=1)  # (num_envs, 2, obs_i)

    def compute_reward(self):
        """Compute rewards for heterogeneous agents."""
        # Can be same or different per agent
        # HAPPO handles both!
        pass
```

#### 3.2 Update Wrapper

**File**: `mqe/envs/wrappers/heterogeneous_push_wrapper.py`

```python
class HeterogeneousPushWrapper:
    """Wrapper for heterogeneous pushing task."""

    def __init__(self, env):
        self.env = env
        self.num_agents = env.num_agents

        # Agent-specific observation processing
        self.obs_processors = [
            self._process_go1_obs,
            self._process_manipulator_obs
        ]

    def _process_go1_obs(self, obs):
        """Process Go1-specific observations."""
        # Same as current go1_push_mid_wrapper
        return obs

    def _process_manipulator_obs(self, obs):
        """Process manipulator-specific observations."""
        # Different processing for manipulator
        return obs
```

---

## Code Modifications Required

### Minimal Changes (High Priority)

| File | Modification | Difficulty |
|------|-------------|-----------|
| `mqe/envs/base/heterogeneous_robot.py` | **Create new** base class | Medium |
| `mqe/envs/configs/heterogeneous_push_config.py` | **Create new** config | Easy |
| `mqe/envs/heterogeneous/heterogeneous_push.py` | **Create new** environment | Hard |
| `mqe/envs/wrappers/heterogeneous_push_wrapper.py` | **Create new** wrapper | Medium |
| `HARL/harl/envs/mapush/mapush_env.py` | **Add** heterogeneous flag handling | Easy |
| `task/heterogeneous/config.py` | **Create new** task config | Easy |
| `task/heterogeneous/train.sh` | **Create new** training script | Easy |

### No Changes Required (HARL Already Supports!)

✅ `HARL/harl/algorithms/actors/happo.py` - Already handles different obs/action spaces
✅ `HARL/harl/algorithms/critics/` - Centralized critic works with any state space
✅ `HARL/harl/runners/on_policy_ha_runner.py` - Sequential updates work for any agent
✅ Checkpoint saving - Already saves per-agent networks

---

## Testing Strategy

### Stage 1: Simple Heterogeneous Test
**Goal**: Verify basic heterogeneity works

**Setup**:
- Agent 0: Go1 (existing)
- Agent 1: Go1 with **different observation space** (e.g., add dummy sensors)
- Same task: Cuboid pushing

**Success Criteria**:
- Different network sizes created
- Training converges
- Both agents learn

### Stage 2: Kinematically Different Robot
**Goal**: Test with truly different robot

**Setup**:
- Agent 0: Go1 quadruped
- Agent 1: Simple wheeled robot (easier than manipulator)
  - 2 DOFs: [left_wheel_vel, right_wheel_vel]
  - Simple observations: [x, y, theta, target_x, target_y]

**Success Criteria**:
- Both robots move in simulation
- Collaborative pushing works
- Training stable

### Stage 3: Manipulator Integration
**Goal**: Full heterogeneous system

**Setup**:
- Agent 0: Go1 quadruped (pushing)
- Agent 1: Dual-arm manipulator (grasping & pulling)

**Success Criteria**:
- Different behaviors emerge
- Performance ≥ homogeneous baseline
- Generalizes to different objects

---

## Timeline Estimate

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| **Phase 1** | Architecture Redesign | 5-7 days | - |
| 1.1 | Create HeterogeneousRobot base class | 2 days | - |
| 1.2 | Agent-specific configuration system | 1 day | 1.1 |
| 1.3 | Multi-asset loading | 1 day | 1.1 |
| 1.4 | Variable DOF handling | 1 day | 1.1 |
| 1.5 | Testing with dummy agents | 1 day | 1.2-1.4 |
| **Phase 2** | HARL Integration | 2-3 days | Phase 1 |
| 2.1 | Update MAPush wrapper | 1 day | 1.5 |
| 2.2 | Test with Stage 1 (simple hetero) | 1-2 days | 2.1 |
| **Phase 3** | Wheeled Robot | 3-4 days | Phase 2 |
| 3.1 | Create wheeled robot URDF | 1 day | - |
| 3.2 | Implement wheeled controller | 1 day | 3.1 |
| 3.3 | Training & debugging | 1-2 days | 3.2 |
| **Phase 4** | Manipulator Integration | 5-7 days | Phase 3 |
| 4.1 | Obtain/create manipulator URDF | 1 day | - |
| 4.2 | Implement IK controller | 2 days | 4.1 |
| 4.3 | Training & tuning | 2-4 days | 4.2 |
| **Total** | **15-21 days** (~3-4 weeks) | | |

---

## Risk Assessment

### High Risk 🔴

1. **Isaac Gym Multi-Asset Handling**
   - Risk: Isaac Gym may have issues with very different robot morphologies in same scene
   - Mitigation: Start with similar morphologies (Go1 + quadruped variant)

2. **Reward Shaping for Heterogeneous Agents**
   - Risk: Agents with different capabilities need different reward functions
   - Mitigation: Careful reward design, extensive ablation studies

### Medium Risk 🟡

3. **Training Stability**
   - Risk: Different learning speeds per agent type
   - Mitigation: Per-agent learning rates, curriculum learning

4. **State Space Design**
   - Risk: Centralized critic needs informative global state
   - Mitigation: Include all agents' proprioception + environment state

### Low Risk 🟢

5. **HARL Compatibility**
   - Risk: HARL might not handle heterogeneity well
   - Assessment: ✅ HARL is **designed for this!** Very low risk.

6. **Checkpoint Management**
   - Risk: Different network sizes complicate saving/loading
   - Assessment: ✅ Already handled by HARL's per-agent saving

---

## Advantages of HAPPO for This Task

### Why HAPPO > MAPPO for Heterogeneous Robots:

| Feature | MAPPO | HAPPO |
|---------|-------|-------|
| Different obs spaces | ❌ Requires padding/masking | ✅ Native support |
| Different action spaces | ❌ Requires dummy actions | ✅ Native support |
| Different network sizes | ❌ Forces same architecture | ✅ Per-agent networks |
| Parameter sharing | ✅ Required (efficiency) | ❌ Optional (flexibility) |
| Sequential updates | ❌ Simultaneous only | ✅ Handles asymmetry |
| Theoretical guarantees | Limited | ✅ Monotonic improvement |

**Bottom Line**: HAPPO was **literally designed** for this exact use case! 🎯

---

## Example: Go1 + Dual-Arm Manipulator

### Robot Specifications

**Agent 0: Unitree Go1**
- Type: Quadruped
- DOFs: 12 (3 per leg)
- Action: (3,) velocity commands [vx, vy, vyaw]
- Observation: (8,) [target_dist, angle, box_pos, partner_pos]
- Role: Push from behind

**Agent 1: Dual-Arm Manipulator**
- Type: Fixed-base manipulator
- DOFs: 14 (7 per arm)
- Action: (12,) joint positions (6 per arm, ignore wrist)
- Observation: (20,) [target_pos, box_pos, ee_poses, joint_states]
- Role: Pull from front / lift

### Task Design

```
┌─────────────────────────────────────┐
│         Target Area (green)         │
└─────────────────────────────────────┘
                 ↑
                 │ Pull
           ┌─────────┐
           │  Box    │  ← Manipulator arms grasping
           └─────────┘
                 ↑
                 │ Push
              [Go1]
```

### Reward Structure

**Shared Rewards** (both agents):
- Distance to goal: `+0.01 × Δd`
- Success: `+10` when box in target
- Collision: `-5` if agents collide

**Agent-Specific Rewards**:
- Go1: `+0.005 × push_force` (encourage pushing)
- Manipulator: `+0.01 × grasp_quality` (encourage stable grasp)

---

## Next Steps

### Before Starting Implementation:

1. ✅ **Get HAPPO baseline** (current 2×Go1) to >90% success rate
2. ✅ **Decide on second robot type**:
   - Easy: Wheeled robot (differential drive)
   - Medium: Different quadruped (ANYmal, Spot)
   - Hard: Manipulator (dual-arm, humanoid)
3. ✅ **Obtain/create URDF** for second robot
4. ✅ **Define task requirements**:
   - What should each robot do?
   - How should they coordinate?
   - What are success criteria?

### Implementation Order:

1. Start with **Stage 1** (simple heterogeneous test)
2. Move to **Stage 2** (wheeled robot) for proof-of-concept
3. Finally **Stage 3** (manipulator) for full heterogeneous system

---

## Conclusion

### Feasibility: ✅ **HIGHLY FEASIBLE**

**Why it's achievable:**
1. HAPPO is **designed** for heterogeneous agents
2. HARL framework **already supports** different obs/action spaces
3. Isaac Gym **can handle** multiple robot types in same scene
4. Architecture changes are **localized** to environment code

### Estimated Effort: **3-4 weeks** (15-21 days)

### Key Success Factors:
1. ✅ Baseline HAPPO performance established
2. ✅ Clear task design with complementary robot roles
3. ✅ Careful reward shaping for heterogeneous capabilities
4. ✅ Incremental testing (simple → complex)

### Recommended Approach:
**Start simple, iterate quickly:**
- Week 1: Architecture + simple heterogeneous test
- Week 2: Wheeled robot integration + training
- Week 3-4: Manipulator integration + tuning

**This is exactly what HAPPO was made for!** 🚀

---

**Document Status**: 📋 Planning Complete - Ready for Review
**Next Action**: Approve plan → Begin Phase 1 implementation
**Last Updated**: 2025-11-15

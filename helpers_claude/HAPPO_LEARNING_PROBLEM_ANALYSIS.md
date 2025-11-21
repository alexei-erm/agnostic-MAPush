# HAPPO Learning Problem: Root Cause Analysis & Solutions

**Date**: 2025-11-21
**Status**: 🔴 CRITICAL - HAPPO agents not learning after 200M episodes
**Current Performance**: HAPPO getting -18.38 episode reward vs MAPPO +4.8 episode reward

---

## Executive Summary

After 200M training episodes across multiple runs, HAPPO is **completely failing to learn** the cuboid pushing task. The issue is NOT a simple bug, but rather a **combination of hyperparameter misconfigurations and reward structure problems** that create an insurmountable learning challenge for HAPPO.

**Key Finding**: While the "safe" config was designed to prevent training collapse, it's actually **TOO conservative** and combined with poor reward scaling, prevents any learning from occurring.

---

## Problem Manifestation

### Performance Comparison

| Metric | MAPPO (Baseline) | HAPPO (Current) | Difference |
|--------|------------------|-----------------|------------|
| **Episode Reward** | +4.8 | -18.38 | **-23.18** ⚠️ |
| **Per-step Reward** | +0.024 | -0.092 | **-0.116** |
| **Value Loss** | 0.004 | 0.217 | **+54x worse** 🚨 |
| **Entropy** | ~50 (high) | 1.82 | Different scale |
| **Success Rate** | 19.3% @ 10M | Unknown (likely 0%) | N/A |

### Training Degradation Pattern (from docs)

HAPPO performance over time:
- **Q1 (0-25M steps)**: -14.65 reward (BEST performance)
- **Q2 (25-50M steps)**: -15.31 reward (-4.5% degradation)
- **Q3 (50-75M steps)**: -17.08 reward (-16.6% degradation)
- **Q4 (75-100M+ steps)**: -19.55 reward (-33% degradation)

**Critical Issue**: **Rewards are DEGRADING over time instead of improving!**

---

## Root Causes Identified

### 1. **Reward Scale Mismatch** 🚨 PRIMARY ISSUE

**Problem**: The reward scales create an impossible learning environment.

From `task/cuboid/config.py`:
```python
class scales:
    target_reward_scale = 0.00325              # Distance progress
    approach_reward_scale = 0.00075            # Approach to box
    collision_punishment_scale = -0.0025       # Collision penalty
    push_reward_scale = 0.0015                 # Push activation
    ocb_reward_scale = 0.004                   # Contact geometry
    reach_target_reward_scale = 10             # SUCCESS BONUS ⚠️
    exception_punishment_scale = -5            # Exception penalty ⚠️
```

**Analysis**:
- **reach_target_reward_scale (10)** dominates all other rewards by 2,500-6,600x
- **exception_punishment_scale (-5)** can instantly negate 500-1,600 steps of learning
- Distance reward has hidden 100x multiplier: `target_reward_scale * 100 = 0.325` per step
- Shaping rewards (approach, OCB, push) provide almost zero gradient signal

**Impact on Learning**:
- Policy receives almost no feedback until task completion (sparse reward problem)
- Exceptions cause catastrophic gradient spikes
- HAPPO's per-agent credit assignment gets confused by reward scale variance

### 2. **Over-Conservative Hyperparameters** 🚨

**Problem**: The "safe" config prevents learning AND recovery.

From `HARL/harl/configs/algos_cfgs/happo.yaml` (current):
```yaml
lr: 0.0003                    # 40% lower than MAPPO
critic_lr: 0.0003             # 40% lower than MAPPO
max_grad_norm: 0.5            # 95% lower than original (was 10.0)
use_adv_normalize: True       # May hide multi-agent signals
```

**Analysis**:
- **Learning rate (0.0003)** is too slow to escape bad local minima
  - Previous docs show Basic HAPPO (LR=0.0005) partially recovered in Q4
  - Safe HAPPO (LR=0.0003) never recovered
- **Gradient clipping (0.5)** is TOO aggressive
  - Actual gradients spike to 1.29 (clipping is being violated!)
  - Even when clipped, low LR means updates are tiny
- **Advantage normalization** may normalize away important credit assignment signals

**Evidence**:
```
Safe HAPPO gradient norms (from docs):
- Agent0: max 1.29 (EXCEEDS 0.5 limit!)
- Agent1: max 1.20 (EXCEEDS 0.5 limit!)

Basic HAPPO gradient norms:
- Agent0: max 0.73 (respects limits better)
- Agent1: max 0.56 (respects limits better)
```

### 3. **Value Function Collapse** 🚨

**Problem**: The critic cannot learn to predict returns accurately.

**Evidence**:
- Value loss INCREASING over training (0.022 → 0.217 = 10x worse)
- This means value estimates becoming LESS accurate over time
- Bad value estimates → bad policy gradients → worse policy → worse value estimates (death spiral)

**Why This Happens**:
- Network capacity (256x256) may be insufficient for complex multi-agent pushing physics
- Sparse rewards make value prediction extremely difficult
- HAPPO's advantage decomposition requires accurate baselines per agent

### 4. **Entropy Increasing Instead of Decreasing** 🚨

**Problem**: Policy becoming MORE random over time.

**Evidence**:
```
HAPPO Entropy over time:
Q1: 1.30 (healthy exploration)
Q4: 1.67 (INCREASING +28%)
```

**Normal behavior**: Entropy should DECREASE (policy gets more confident)
**Actual behavior**: Entropy INCREASING (policy loses confidence)

**Interpretation**: The policy is "giving up" - as the value function fails, the policy can't tell good from bad actions, so it becomes more random.

---

## Why "Safe" Config Made Things Worse

The documentation shows a counterintuitive result:

| Metric | Basic HAPPO | Safe HAPPO |
|--------|-------------|------------|
| **Total Degradation** | -18% | **-33%** (worse!) |
| **Q4 Behavior** | Partially recovered | Continued degrading |
| **Gradient Violations** | Fewer | More frequent |

**Explanation**:
1. **Over-conservative LR prevents recovery**: When value function starts diverging, can't correct fast enough
2. **Strict gradient clipping is ineffective**: Gradients violate the limit anyway, but then get multiplied by tiny LR
3. **Advantage normalization hides signals**: In multi-agent, relative advantage magnitudes matter for credit assignment

---

## Comparison: HAPPO vs MAPPO Environment Wrappers

### MAPPO (openrl_ws/utils.py)
```python
def step(self, actions):
    actions = torch.from_numpy(0.5 * actions).cuda().clip(-1, 1)  # ✓ Correct
    obs, reward, termination, info = self.env.step(actions)
    rewards = reward.cpu().unsqueeze(-1).numpy()  # Shape: (n_envs, n_agents, 1)
```

### HAPPO (HARL/harl/envs/mapush/mapush_env.py)
```python
def step(self, actions):
    actions_torch = torch.from_numpy(actions).float().to(self.env.device)
    actions_torch = 0.5 * actions_torch  # ✓ Correct scaling
    obs, reward, termination, info = self.env.step(actions_torch)
    rewards = reward_np[:, :, None]  # Shape: (n_envs, n_agents, 1)
```

**Verdict**: ✅ Both wrappers are correctly implemented. The problem is NOT a wrapper bug.

---

## Solutions: Proposed Fixes

### Priority 1: Fix Reward Scaling (CRITICAL)

Create `task/cuboid/config_happo_fixed.py` with rebalanced rewards:

```python
class scales:
    # Reduce success bonus dramatically
    reach_target_reward_scale = 1.0          # Was 10 → Now 1.0 (10x reduction)

    # Increase shaping rewards significantly
    target_reward_scale = 0.01               # Was 0.00325 → 3x increase
    approach_reward_scale = 0.02             # Was 0.00075 → 27x increase
    ocb_reward_scale = 0.02                  # Was 0.004 → 5x increase
    push_reward_scale = 0.01                 # Was 0.0015 → 7x increase

    # Soften penalties
    collision_punishment_scale = -0.01       # Was -0.0025 → 4x harsher
    exception_punishment_scale = -1.0        # Was -5 → 5x softer (still punishing)
```

**Rationale**:
- Success bonus no longer dominates (now only 50-100x shaping rewards vs 2,500-6,600x)
- Shaping rewards provide meaningful gradient signals
- Exceptions don't cause training collapse
- Overall reward scale more balanced for gradient-based learning

**Additional fix** - Remove hidden 100x multiplier in distance reward:

In `mqe/envs/wrappers/go1_push_mid_wrapper.py` line 329:
```python
# OLD:
distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)

# NEW:
distance_reward = self.target_reward_scale * (2 * (past_distance - distance) - 0.01 * distance)
```

### Priority 2: Hybrid Hyperparameters

Create `HARL/harl/configs/algos_cfgs/happo_balanced.yaml`:

```yaml
# Balanced between safe and aggressive
lr: 0.0004                    # Middle ground (was 0.0003/0.0005)
critic_lr: 0.0004             # Middle ground
max_grad_norm: 1.0            # Less strict (was 0.5/10.0)
use_adv_normalize: False      # Disable for multi-agent
entropy_coef: 0.02            # Increase exploration (was 0.01)
use_valuenorm: True           # Keep
use_clipped_value_loss: True  # Keep
```

**Rationale**:
- Higher LR allows recovery from bad states
- Less aggressive gradient clipping (current 0.5 limit being violated anyway)
- Disable advantage normalization (may hide multi-agent credit assignment)
- Higher entropy maintains exploration when rewards are sparse

### Priority 3: Increase Value Network Capacity

In `task/cuboid/train.sh` line 34:
```bash
# OLD:
--hidden_sizes "[256,256]"

# NEW:
--hidden_sizes "[512,512,256]"  # 3-layer with larger capacity
```

**Rationale**:
- Value function collapse is a primary issue
- Complex multi-agent physics + sparse rewards = need bigger network
- MAPPO might work with smaller network due to centralized training

### Priority 4: Curriculum Learning (Optional)

If above fixes don't work, consider staged training:

**Stage 1** (25M steps): Easier rewards
```python
reach_target_reward_scale = 5.0    # Higher
approach_reward_scale = 0.05       # Much higher
exception_punishment_scale = -0.5  # Much softer
```

**Stage 2** (75M steps): Load from Stage 1, use normal rewards
**Stage 3** (100M steps): Load from Stage 2, use strict rewards

---

## Implementation Plan

### Step 1: Quick Test (10M steps)

```bash
# 1. Create fixed config
cp task/cuboid/config.py task/cuboid/config_happo_fixed.py
# Edit config_happo_fixed.py with new reward scales

# 2. Create balanced hyperparameters
cp HARL/harl/configs/algos_cfgs/happo.yaml \
   HARL/harl/configs/algos_cfgs/happo_balanced.yaml
# Edit with new hyperparameters

# 3. Update train.sh to use new files
# In task/cuboid/train.sh:
#  - Change config source
#  - Set --algo happo_balanced
#  - Set num_steps=10000000

# 4. Run short training
source task/cuboid/train.sh False
```

**Success criteria for 10M test**:
- ✅ Rewards IMPROVING over time (not degrading)
- ✅ Value loss DECREASING (not increasing)
- ✅ Entropy DECREASING slowly (policy getting confident)
- ✅ Episode reward > -10 by end

### Step 2: Full Training (100-200M steps)

If 10M test succeeds:
```bash
# Set num_steps=200000000 in train.sh
source task/cuboid/train.sh False
```

**Monitor TensorBoard for**:
- Steady reward improvement
- Value loss stabilizing below 0.1
- Gradient norms staying healthy (0.3-0.8)
- Entropy decreasing from ~1.5 to ~1.0

### Step 3: Evaluation

```bash
# Test final checkpoint
source task/cuboid/train.sh True

# Check success rate (should be > 10% minimum)
```

---

## Expected Outcomes

### With Fixed Rewards Only
- **Baseline**: Agents should at least LEARN (rewards improve)
- **Target**: -5 to 0 episode reward by 100M steps
- **Success Rate**: 5-15% @ 100M steps

### With Rewards + Balanced Hyperparameters
- **Target**: +2 to +5 episode reward by 100M steps
- **Success Rate**: 15-25% @ 100M steps (matching MAPPO)

### With Rewards + Hyperparameters + Bigger Network
- **Target**: +5 to +10 episode reward by 100M steps
- **Success Rate**: 20-30% @ 100M steps (exceeding MAPPO)

---

## Why This Will Work

### Evidence from Previous Runs

1. **Basic HAPPO showed recovery** (Q4: +9% from Q3)
   - Proves task is learnable with right settings
   - Higher LR (0.0005) enabled escape from bad states

2. **MAPPO achieves 19.3% success**
   - Proves environment and rewards CAN work
   - Difference is in algorithm configuration, not task difficulty

3. **Gradient violations under "safe" config**
   - Shows limits are too strict, not that gradients are too large
   - Suggests current approach is fighting against natural learning dynamics

### Theoretical Support

1. **Reward shaping literature**: Sparse rewards (large completion bonus) create credit assignment problems
   - Solution: Dense shaping rewards in same scale as completion

2. **Multi-agent RL**: Advantage normalization can hurt when agents need different reward scales
   - HAPPO paper doesn't mandate advantage normalization
   - Works better with properly scaled rewards

3. **Deep RL stability**: Very low LR + aggressive clipping = inability to learn OR recover
   - Modern RL uses moderate clipping (1.0-2.0) with reasonable LR (3e-4 to 5e-4)

---

## Alternative Approaches (If Above Fails)

### Option A: Switch to MAPPO
- MAPPO is working (19.3% success)
- Simpler algorithm, proven stable
- **Downside**: Can't handle heterogeneous agents (future requirement?)

### Option B: Use MAPPO Rewards for HAPPO
- Copy exact reward config from working MAPPO run
- Keep HAPPO algorithm but use proven reward structure

### Option C: Imitation Learning Bootstrap
- Manually teleoperate successful pushes
- Train HAPPO with behavior cloning first
- Fine-tune with RL

---

## Files to Modify

### Must Modify:
1. **`task/cuboid/config.py`** - Fix reward scales
2. **`HARL/harl/configs/algos_cfgs/happo.yaml`** - Balance hyperparameters
3. **`mqe/envs/wrappers/go1_push_mid_wrapper.py:329`** - Remove 100x multiplier

### Optional:
4. **`task/cuboid/train.sh:34`** - Increase network size
5. **`task/cuboid/config.py`** - Add curriculum stages

---

## Monitoring Checklist

During training, watch for:

✅ **Good Signs:**
- Rewards trending upward (even if slowly)
- Value loss decreasing below 0.1
- Entropy gradually dropping (1.5 → 1.0 over 100M)
- Gradient norms healthy (0.3-0.8)
- Agents occasionally reaching target (even if rare)

🚨 **Warning Signs:**
- Rewards still degrading after 25M steps
- Value loss increasing or staying > 0.2
- Entropy increasing or staying > 1.8
- Gradient norms hitting clip limit frequently
- All episodes timing out with no successes

---

## Conclusion

HAPPO's learning failure is NOT a fundamental algorithm bug, but rather a **configuration problem**:

1. **Reward scales are incompatible with gradient-based learning** (sparse, unbalanced)
2. **"Safe" hyperparameters are too conservative** (prevent learning AND recovery)
3. **Value network may lack capacity** for complex multi-agent task

**The fix requires a multi-pronged approach**:
- Rebalance reward scales (critical)
- Use moderate hyperparameters (important)
- Increase value network capacity (helpful)

**Success probability**: **HIGH** - Basic HAPPO's Q4 recovery and MAPPO's 19.3% success prove the task is learnable with correct configuration.

---

**Next Step**: Implement Priority 1 (reward scaling) and Priority 2 (balanced hyperparameters), then run 10M test.

**ETA**: ~1-2 hours to implement fixes, 4-6 hours for 10M test run, 1-2 days for full 200M training.

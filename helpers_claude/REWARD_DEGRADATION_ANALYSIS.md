# CRITICAL: HAPPO Reward Degradation Analysis

**Date**: 2025-11-19
**Status**: 🚨 BOTH MODELS SHOWING SEVERE PERFORMANCE DEGRADATION
**Severity**: HIGH - Training is NOT learning properly

---

## TL;DR - The Problem

**BOTH configurations (Safe and Basic HAPPO) are experiencing severe reward degradation:**

- **Safe HAPPO**: Rewards degraded **-33.4%** (from -14.65 to -19.55)
- **Basic HAPPO**: Rewards degraded **-18.2%** (from -15.00 to -17.73)

**This means the agents are getting WORSE at the task over time, not better.**

---

## Detailed Reward Analysis

### SAFE HAPPO (Current Run - 59M steps)

| Quartile | Step Range | Mean Reward | Change from Q1 |
|----------|------------|-------------|----------------|
| **Q1 (0-25%)** | 0.5M - 14.5M | **-14.65** | baseline |
| **Q2 (25-50%)** | 15M - 29.5M | **-15.31** | -0.65 (-4.5%) |
| **Q3 (50-75%)** | 30M - 44M | **-17.08** | -2.43 (-16.6%) ⚠️ |
| **Q4 (75-100%)** | 44.5M - 59M | **-19.55** | -4.90 (-33.4%) 🚨 |

**Trajectory**: Continuous degradation, accelerating in later stages

### BASIC HAPPO (Completed - 100M steps)

| Quartile | Step Range | Mean Reward | Change from Q1 |
|----------|------------|-------------|----------------|
| **Q1 (0-25%)** | 0.5M - 25M | **-15.00** | baseline |
| **Q2 (25-50%)** | 25.5M - 50M | **-16.74** | -1.74 (-11.6%) ⚠️ |
| **Q3 (50-75%)** | 50.5M - 75M | **-19.49** | -4.49 (-29.9%) 🚨 |
| **Q4 (75-100%)** | 75.5M - 100M | **-17.73** | -2.73 (-18.2%) |

**Trajectory**: Degraded severely, then **partially recovered** in Q4

**Note**: Basic HAPPO's Q4 improvement (+1.77 from Q3) suggests some self-correction, but still significantly worse than Q1.

---

## Quarter-to-Quarter Comparison

### Safe HAPPO Degradation Pattern
```
Q1 → Q2: -0.65  (-4.5%)   ⚠️ Minor decline
Q2 → Q3: -1.77  (-11.6%)  🚨 Accelerating
Q3 → Q4: -2.47  (-14.5%)  🚨 Severe collapse
```

### Basic HAPPO Degradation Pattern
```
Q1 → Q2: -1.74  (-11.6%)  🚨 Sharp drop
Q2 → Q3: -2.75  (-16.5%)  🚨 Worst period
Q3 → Q4: +1.77  (+9.1%)   ✅ Partial recovery
```

---

## Critical Metrics Analysis

### Gradient Norms

| Metric | Safe HAPPO | Basic HAPPO | Analysis |
|--------|------------|-------------|----------|
| **Agent0 Max Grad** | 1.29 | 0.73 | Safe has HIGHER spikes despite lower clip! 🚨 |
| **Agent1 Max Grad** | 1.20 | 0.56 | Safe has HIGHER spikes despite lower clip! 🚨 |
| **Grad Trend** | Decreasing | Decreasing | Both stabilizing (good) |

**CRITICAL FINDING**: Safe HAPPO's `max_grad_norm=0.5` is being **violated** (spikes to 1.29). This suggests:
- Clipping not working as expected
- OR gradient explosion happening despite clipping
- OR framework not respecting the clip parameter

### Value Loss (Critic Performance)

| Metric | Safe HAPPO | Basic HAPPO | Analysis |
|--------|------------|-------------|----------|
| **Q1 Mean** | 0.022 | 0.025 | Both low (good) |
| **Q4 Mean** | 0.061 | 0.130 | Basic degraded more |
| **Trend** | +0.038 | +0.105 🚨 | Basic critic struggling more |
| **Max Loss** | 0.128 | 0.247 🚨 | Basic had severe spikes |

**Analysis**:
- Value loss **increasing** in both models = critic getting worse at predicting returns
- Basic HAPPO's critic degraded more severely (2.7x increase vs 1.7x in Safe)
- This explains why Basic recovered in Q4: critic finally caught up

### Entropy (Exploration)

| Metric | Safe HAPPO | Basic HAPPO | Analysis |
|--------|------------|-------------|----------|
| **Q1 Mean** | 1.30 | 1.40 | Healthy exploration |
| **Q4 Mean** | 1.67 | 1.93 | **INCREASING** (unusual) |
| **Trend** | +0.37 | +0.52 | Both getting MORE random 🚨 |

**CRITICAL FINDING**: Entropy is **INCREASING**, not decreasing!

Normal behavior: Entropy should **decrease** over training (policy becomes more confident)

Current behavior: Entropy **increasing** = policy becoming MORE random/uncertain

**This is a red flag** - the policy is losing confidence instead of gaining it.

---

## Root Cause Analysis

### Why Are Rewards Getting Worse?

Based on the metrics, here are the likely causes:

#### 1. **Value Function Collapse** 🚨 PRIMARY SUSPECT

**Evidence**:
- Value loss increasing in both models
- Rewards degrading as value loss increases
- Pattern matches "value divergence" problem

**What's happening**:
- Critic (value function) is unable to accurately predict returns
- As critic gets worse, actor gets bad gradient signals
- Actor's policy degrades because it's following misleading value estimates

**Why**:
- Task may be too complex for current value network architecture
- Non-stationary environment (changing object dynamics)
- Insufficient value network capacity (256x256 may be too small)

#### 2. **Exploration-Exploitation Imbalance** 🚨

**Evidence**:
- Entropy INCREASING instead of decreasing
- Agents becoming more random over time
- Suggests agents are "giving up" on consistent strategy

**What's happening**:
- Policy is becoming less deterministic (more random)
- This is opposite of expected learning behavior
- Suggests policy is unstable or overfitting then reverting

**Why**:
- `entropy_coef=0.01` may be too low initially, then causing oscillation
- Policy may be hitting local minima then escaping randomly
- Multi-agent credit assignment problem (who caused the failure?)

#### 3. **Multi-Agent Coordination Collapse** 🚨

**Evidence**:
- Two agents (Go1 + manipulator) must coordinate
- Individual entropy increasing for BOTH agents
- Rewards degrading despite individual metrics looking "healthy"

**What's happening**:
- Each agent's policy looks OK individually
- But coordination between agents is breaking down
- Classic multi-agent problem: independent learning in non-stationary environment

**Why**:
- As one agent's policy changes, the other agent's world becomes non-stationary
- HAPPO's advantage decomposition may not be sufficient
- Need stronger coordination mechanism

#### 4. **Reward Sparsity / Difficult Task**

**Evidence**:
- Initial rewards already very negative (-6 to -7)
- Never achieving positive rewards
- Continuous degradation suggests task is too hard

**What's happening**:
- Agents never finding successful strategy
- Initial "success" (Q1) may just be lucky random behavior
- As they "learn", they're actually moving away from lucky initialization

**Why**:
- Task might be beyond current approach (pushing cuboid is hard)
- Reward shaping may be insufficient
- Object physics may be too unstable

---

## Comparison: Safe vs Basic

### Safe HAPPO Performance (Current)

**Advantages**:
✅ Lower value loss (0.13 vs 0.25 max)
✅ More controlled gradient norms
✅ Gradients decreasing (stabilizing)

**Disadvantages**:
❌ WORSE overall degradation (-33% vs -18%)
❌ No recovery in Q4 (still degrading)
❌ Gradient clipping being violated (spikes to 1.29 with 0.5 limit)

### Basic HAPPO Performance

**Advantages**:
✅ Less severe overall degradation (-18% vs -33%)
✅ Showed recovery in Q4 (+9%)
✅ Respected gradient limits better

**Disadvantages**:
❌ Higher value loss spikes (0.25)
❌ Still significantly degraded vs Q1
❌ More unstable value learning

### Verdict

**NEITHER configuration is working well**, but:
- **Basic HAPPO is currently performing better** (less degradation, showed recovery)
- **Safe HAPPO's conservative settings may be TOO conservative**, preventing recovery
- The "safe" config's restrictions may be preventing the model from escaping bad states

---

## Why Safe HAPPO Is Doing Worse

This is counterintuitive but explainable:

### 1. **Over-Conservative Learning Rate**
- LR of 3e-4 may be too slow to escape bad local minima
- When value function starts diverging, can't correct fast enough
- Basic's 5e-4 LR allowed it to recover in Q4

### 2. **Advantage Normalization Backfiring**
- `use_adv_normalize=True` may be normalizing away important signal
- In multi-agent setting, relative advantage magnitudes matter
- Normalization may be hiding which agent is responsible for failure

### 3. **Too-Strict Gradient Clipping**
- Despite max_grad_norm=0.5, gradients still spike to 1.29
- This suggests the clipping isn't the actual problem
- Lower learning rate means these (already clipped) gradients have even less effect
- Result: Can't learn OR correct mistakes

---

## What's Actually Happening (Hypothesis)

### The Degradation Cycle

```
1. Agents start with random initialization
   ↓
2. Get lucky with early random behavior (Q1: -14 to -15 reward)
   ↓
3. Start "learning" from these lucky experiences
   ↓
4. Critic tries to predict value but task is too hard
   ↓
5. Critic's value estimates become increasingly wrong
   ↓
6. Actor gets bad gradient signals from wrong value estimates
   ↓
7. Actor's policy degrades (Q2-Q3: performance collapse)
   ↓
8. DIVERGENCE POINT:
   - Basic HAPPO: Higher LR allows relearning (Q4 recovery)
   - Safe HAPPO: Too conservative, can't escape (Q4 continues degrading)
```

### Why Entropy Increases

As the value function becomes less reliable:
- Policy gradient becomes noisier
- Actor can't tell what's actually good/bad
- Entropy increases = policy becomes more uncertain
- This is a **failure mode**, not healthy exploration

---

## Recommendations

### Immediate Actions (Stop current training)

🛑 **STOP the current Safe HAPPO run** - it's getting worse and won't recover

The data shows:
- Safe HAPPO at 59M steps: -19.55 reward (worst yet)
- Trend shows continued degradation
- No signs of recovery like Basic HAPPO showed

**You're wasting compute** running to 100M steps.

### Short-Term Fixes (Try these next)

#### Option 1: Hybrid Configuration ⭐ RECOMMENDED

Create `happo_hybrid.yaml`:

```yaml
# Take best of both worlds
lr: 0.0004                    # Middle ground (was 0.0003/0.0005)
critic_lr: 0.0004             # Middle ground
max_grad_norm: 1.0            # Less strict (was 0.5/10.0)
use_adv_normalize: False      # Disable (may be harmful in multi-agent)
entropy_coef: 0.02            # INCREASE (was 0.01) - maintain exploration
use_valuenorm: True           # Keep
use_clipped_value_loss: True  # Keep
```

**Rationale**:
- Higher LR allows recovery from bad states
- Disable advantage normalization (may be hiding multi-agent signals)
- Increase entropy to maintain exploration
- Less strict gradient clipping (current strict limit being violated anyway)

#### Option 2: Increase Value Network Capacity

Current: `hidden_sizes: [256, 256]`
Proposed: `hidden_sizes: [512, 512, 256]` or `[1024, 512]`

**Rationale**:
- Value function collapse is primary issue
- Bigger network may handle complex multi-agent value estimation
- Cuboid pushing requires predicting complex physics

#### Option 3: Adjust Reward Shaping

The task may be too hard. Consider:
- Add intermediate rewards (distance to cuboid, contact with cuboid, etc.)
- Reduce penalties
- Add curriculum learning (start with easier objects)

### Medium-Term Solutions

#### 1. **Curriculum Learning**
```
Stage 1: Push cylinder (easiest) - 25M steps
Stage 2: Push cuboid (current task) - load from Stage 1
Stage 3: Push irregular objects
```

#### 2. **Communication Between Agents**
- Add explicit communication channel between Go1 and manipulator
- Share observations / intentions
- May improve coordination

#### 3. **Different Algorithm**
Consider switching from HAPPO to:
- **MAPPO**: Simpler, may work better for this task
- **QMIX**: Better for coordination
- **MAT** (Multi-Agent Transformer): State-of-the-art for multi-agent

### Long-Term Investigation

#### 1. **Check Environment**
- Is the cuboid physics stable?
- Are there any bugs in reward calculation?
- Is the task actually solvable?

#### 2. **Baseline Testing**
Test with single agent (just Go1) on simpler task:
- If single agent also degrades → problem with base algorithm
- If single agent succeeds → multi-agent coordination is the issue

#### 3. **Expert Demonstrations**
- Manually teleoperate the robots to push cuboid successfully
- Use imitation learning to bootstrap policy
- Then fine-tune with RL

---

## Action Plan

### Step 1: Stop Current Training ✋
```bash
# Kill the running Safe HAPPO training
pkill -f "happo-mapush-cuboid"
```

### Step 2: Create Hybrid Config 📝

```bash
# Create new config
cp HARL/harl/configs/algos_cfgs/happo_safe.yaml \
   HARL/harl/configs/algos_cfgs/happo_hybrid.yaml

# Edit with recommendations above
```

### Step 3: Run Short Test (10M steps) 🧪

```bash
# Test hybrid config for 10M steps
# Edit train.sh: algo="happo_hybrid", num_steps=10000000
source task/cuboid/train.sh False
```

### Step 4: Monitor Closely 👀

Watch TensorBoard for:
- ✅ Value loss: Should DECREASE or stay flat, not increase
- ✅ Entropy: Should DECREASE slowly (1.3 → 1.0), not increase
- ✅ Rewards: Should IMPROVE or plateau, not degrade

### Step 5: Decide Next Steps

**If hybrid improves** (rewards better than -15 after 10M):
→ Continue to 100M steps

**If hybrid also degrades**:
→ Problem is deeper (task difficulty, environment, or algorithm choice)
→ Try medium-term solutions (bigger network, curriculum, different algorithm)

---

## Key Insights Summary

### What We Learned

1. **"Safe" doesn't mean "better"** - Over-conservative hyperparameters can prevent learning AND recovery

2. **Value function is the bottleneck** - Both models showed value loss increasing as rewards degraded

3. **Multi-agent coordination is hard** - Increasing entropy suggests agents can't find stable coordination strategy

4. **Basic HAPPO's recovery** (Q4) shows the task is learnable with right settings

5. **Gradient clipping violations** - Safe config's 0.5 limit was exceeded (1.29), suggesting limits aren't the core issue

### What Doesn't Work

❌ Very low learning rate (3e-4) - too slow to recover
❌ Very strict gradient clipping (0.5) - gets violated anyway, and prevents learning
❌ Advantage normalization in multi-agent - may hide important signals
❌ Current entropy coefficient (0.01) - too low, entropy still increasing

### What Might Work

✅ Moderate learning rate (4e-4 to 5e-4)
✅ Moderate gradient clipping (1.0 to 2.0)
✅ Higher entropy coefficient (0.02)
✅ Bigger value network (512x512 or 1024x512)
✅ Curriculum learning (start easier)

---

## Conclusion

**Both HAPPO configurations are failing**, but for different reasons:

- **Safe HAPPO**: Too conservative, can't recover from mistakes
- **Basic HAPPO**: More unstable, but showed ability to recover

The **root cause** is likely:
1. Value function collapse (primary)
2. Multi-agent coordination difficulty (secondary)
3. Task complexity / reward sparsity (tertiary)

**Recommended path forward**:
1. Stop current Safe HAPPO training (wasting compute)
2. Try hybrid configuration with balanced hyperparameters
3. If hybrid fails, increase value network capacity
4. If still failing, consider curriculum learning or different algorithm

**The good news**: Basic HAPPO's Q4 recovery proves the task is learnable with the right approach.

---

**Files**:
- This analysis: `REWARD_DEGRADATION_ANALYSIS.md`
- Training comparison: `HAPPO_TRAINING_ANALYSIS.md`
- Config guide: `helpers_claude/HAPPO_SAFE_CONFIG_GUIDE.md`

**Created**: 2025-11-19 16:10
**Next Review**: After hybrid config 10M step test

# HAPPO Safe Configuration Guide

**Created**: 2025-11-18
**Purpose**: Stabilize HAPPO training for long runs (100M+ steps)

---

## Problem Identified

Your HAPPO training showed performance degradation:
- **First 25%**: -15.00 reward (best)
- **Third 25%**: -19.49 reward (worst) ⚠️
- **Last 25%**: -17.73 reward (partial recovery)

This pattern suggests **training instability** around 50M steps, likely caused by:
1. **Gradient explosion** (max_grad_norm too high)
2. **Unstable advantage estimation** (no normalization)

---

## Configuration Comparison

| Parameter | Original happo.yaml | **happo_safe.yaml** | Change Reason |
|-----------|---------------------|---------------------|---------------|
| **lr** | 5e-4 | **3e-4** ⬇️ | Prevent divergence in long runs |
| **critic_lr** | 5e-4 | **3e-4** ⬇️ | Match actor LR for stability |
| **max_grad_norm** | 10.0 | **0.5** ⬇️⚠️ | **CRITICAL**: Prevent gradient explosion |
| **use_adv_normalize** | ❌ Not set | **True** ✅ | Robust to value errors |
| **use_valuenorm** | True ✓ | True ✓ | Keep (already good) |
| **entropy_coef** | 0.01 ✓ | 0.01 ✓ | Keep (exploration) |
| **clip_param** | 0.2 ✓ | 0.2 ✓ | Keep (standard PPO) |
| **share_param** | False ✓ | False ✓ | Keep (HAPPO-specific) |
| **fixed_order** | False ✓ | False ✓ | Keep (HAPPO-specific) |

---

## How to Use

### Option 1: Use safe config by default

Edit the train script to use `happo_safe` instead of `happo`:

```bash
# In task/cuboid/train.sh, change:
python HARL/examples/train.py --algo happo_safe ...
```

### Option 2: Modify train.sh with a flag

You can add a variable to switch between configs:

```bash
# Add at top of train.sh
USE_SAFE_HAPPO=true  # Set to false for original config

if [ "$USE_SAFE_HAPPO" = true ]; then
    happo_algo="happo_safe"
else
    happo_algo="happo"
fi

# Then use:
python HARL/examples/train.py --algo "$happo_algo" ...
```

### Option 3: Command line override

You can also override specific parameters without changing configs:

```bash
python HARL/examples/train.py \
    --algo happo \
    --env mapush \
    --max_grad_norm 0.5 \
    --lr 0.0003 \
    --critic_lr 0.0003 \
    --use_adv_normalize True
```

---

## Expected Benefits

### With Safe Config:

✅ **More stable training**
- Gradients won't explode due to max_grad_norm=0.5
- Advantage normalization prevents outlier episodes from destabilizing policy

✅ **Better long-term performance**
- Lower learning rate prevents overshooting optimal policy
- Should maintain good performance throughout 100M steps

✅ **Smoother learning curves**
- Less variance in episode rewards
- More consistent improvement

### Trade-offs:

⚠️ **Slower initial learning**
- Lower LR means slower convergence early on
- Might take longer to reach good performance initially

⚠️ **Potentially lower peak performance** (in some cases)
- Conservative gradients might miss some optimization opportunities
- But more likely to maintain performance in long runs

---

## Testing Recommendations

### 1. Quick Test (10M steps)
```bash
# Edit task/cuboid/train.sh:
# - Set algo="happo_safe"
# - Set num_steps=10000000

source task/cuboid/train.sh False
```

Check after 10M steps:
- Are rewards improving smoothly?
- Is entropy staying healthy (>1.0)?
- Are gradients stable?

### 2. Compare with Original

Run **both** configs side-by-side:

**Terminal 1**: Original HAPPO
```bash
python HARL/examples/train.py --algo happo --env mapush --exp_name cuboid_original --seed 1
```

**Terminal 2**: Safe HAPPO
```bash
python HARL/examples/train.py --algo happo_safe --env mapush --exp_name cuboid_safe --seed 1
```

Then compare TensorBoard:
```bash
tensorboard --logdir results/mapush/go1push_mid/
```

### 3. Full Run (100M steps)

If quick test looks good, run full training:
```bash
# Set in train.sh:
# - algo="happo_safe"
# - num_steps=100000000

source task/cuboid/train.sh False
```

Monitor for:
- No performance degradation around 50M steps
- Steady improvement or plateau (not decline)
- Value loss staying bounded

---

## Monitoring Checklist

### During Training, Watch For:

✅ **Good Signs:**
- Rewards improving or staying stable
- Entropy gradually decreasing (1.5 → 1.0 over 100M steps)
- Value loss < 1.0
- Grad norms staying under 0.5
- No sudden reward drops

⚠️ **Warning Signs:**
- Rewards dropping suddenly (>2 points)
- Entropy dropping below 0.5 (policy too deterministic)
- Value loss exploding (>10.0)
- Grad norms hitting 0.5 frequently
- NaN in any metric

### TensorBoard Metrics to Monitor:

1. **train_episode_rewards**: Should improve or plateau
2. **agent0/dist_entropy**: Should stay >0.5
3. **agent1/dist_entropy**: Should stay >0.5
4. **critic/value_loss**: Should stay <1.0
5. **agent0/actor_grad_norm**: Should stay <0.5
6. **agent1/actor_grad_norm**: Should stay <0.5

---

## Troubleshooting

### If training is too slow:
```yaml
# Increase LR slightly (but stay safe)
lr: 0.0005  # Instead of 0.0003
critic_lr: 0.0005
```

### If training is still unstable:
```yaml
# Make even safer
max_grad_norm: 0.3  # Instead of 0.5
lr: 0.0002  # Even lower
```

### If rewards plateau early:
```yaml
# Increase exploration
entropy_coef: 0.02  # Instead of 0.01
```

---

## Comparison with MAPPO Safe Config

The `happo_safe.yaml` is designed to match the stability of `ppo_safe.yaml`:

| Feature | ppo_safe.yaml | happo_safe.yaml |
|---------|---------------|-----------------|
| Learning rate | 3e-4 ✓ | 3e-4 ✓ |
| Max grad norm | 0.5 ✓ | 0.5 ✓ |
| Adv normalize | True ✓ | True ✓ |
| Value norm | True ✓ | True ✓ |
| Clip param | 0.2 ✓ | 0.2 ✓ |
| **Unique to HAPPO** | N/A | share_param: False, fixed_order: False |

This ensures HAPPO has the **same stability guarantees** as the proven safe MAPPO config.

---

## Files Modified

1. **Created**: `HARL/harl/configs/algos_cfgs/happo_safe.yaml`
   - Safe configuration file with comments explaining each change

2. **Reference**: `openrl_ws/cfgs/ppo_safe.yaml`
   - OpenRL safe config that inspired these changes

3. **Original**: `HARL/harl/configs/algos_cfgs/happo.yaml`
   - Kept unchanged for comparison

---

## Next Steps

1. ✅ **happo_safe.yaml created** with safer parameters
2. ⏭️ **Test with short run** (10M steps) to verify stability
3. ⏭️ **Compare with original** side-by-side if needed
4. ⏭️ **Run full 100M training** with safe config
5. ⏭️ **Monitor TensorBoard** for stability improvements

---

**Quick Start:**

```bash
# Edit train.sh line 3:
algo="happo_safe"  # Changed from "happo"

# Then train:
source task/cuboid/train.sh False
```

That's it! Your training should now be more stable for long runs.

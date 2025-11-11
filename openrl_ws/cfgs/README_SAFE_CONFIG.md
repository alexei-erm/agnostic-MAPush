# Safe PPO Configuration

## Overview

`ppo_safe.yaml` is a stability-enhanced version of the standard PPO configuration designed to prevent training crashes in long runs (>80M steps).

## Problem This Solves

### The Issue
During long training runs (>80M steps), the original `ppo.yaml` config can experience:
- **Value function divergence** around 85-90M steps
- Sudden reward collapse (e.g., +0.022 → -0.017 within 5 episodes)
- Value loss explosion (0.004 → 0.4+, a 100x spike)
- Eventually leads to NaN in policy network and crash

**Example:** Run8 (results/11-07-17_cuboid) crashed at episode 973 (97.3M steps) after value divergence began at episode 870 (87M steps).

### Root Cause
1. **No gradient clipping** - Single bad batch can corrupt network parameters
2. **No advantage normalization** - Makes training sensitive to value function errors
3. **High learning rate** - lr=5e-3 is too aggressive for late-stage training
4. **No safeguards** - Missing standard RL stability measures

## Changes from ppo.yaml

### Critical Safety Measures

| Parameter | ppo.yaml | ppo_safe.yaml | Why Changed |
|-----------|----------|---------------|-------------|
| `max_grad_norm` | ❌ Not set | ✅ 0.5 | Prevents gradient explosion that corrupts network |
| `use_adv_normalize` | ❌ false | ✅ true | Makes training robust to value function errors |
| `lr` / `critic_lr` | 5e-3 | 3e-4 | Lower LR prevents overshooting in late training |
| `clip_param` | (default) | 0.2 (explicit) | Limits policy change per update |

### Unchanged Parameters
- `use_valuenorm: true` (kept from original)
- `episode_length: 200`
- `log_interval: 5`
- `use_recurrent_policy: false`
- `use_joint_action_loss: false`

## Usage

### For New Training Runs

Edit your `task/<task>/train.sh`:

```bash
# Change this line:
--config ./openrl_ws/cfgs/ppo.yaml \

# To this:
--config ./openrl_ws/cfgs/ppo_safe.yaml \
```

### For Resuming After Crash

If your training crashed due to value divergence:

```bash
# Resume from checkpoint before the crash (e.g., 80M if crash at 97M)
checkpoint="/path/to/results/checkpoints/rl_model_80000000_steps/module.pt"

python ./openrl_ws/train.py \
    --num_envs 500 \
    --train_timesteps 180000000 \
    --algo mappo \
    --config ./openrl_ws/cfgs/ppo_safe.yaml \  # <-- Use safe config
    --checkpoint $checkpoint \
    --exp_name cuboid \
    --task go1push_mid \
    --use_tensorboard \
    --headless
```

## When to Use Which Config

### Use `ppo.yaml` (original) when:
- Short training runs (<50M steps)
- Quick experiments or prototyping
- You know the training is stable for your specific task

### Use `ppo_safe.yaml` (recommended) when:
- Long training runs (>80M steps) ✅
- Training to 180M steps for full convergence ✅
- You've experienced crashes before ✅
- You want guaranteed stability ✅

## Expected Differences

### Training Speed
- **Slightly slower convergence** initially due to lower learning rate
- **More stable in late training** - no crashes
- **Better final performance** by avoiding collapse

### Metrics to Watch
With `ppo_safe.yaml`, you should see:
- ✅ Gradual, monotonic improvement in rewards
- ✅ Value loss staying < 0.1 throughout training
- ✅ Critic grad norm staying < 1.0
- ✅ No sudden reward drops

If you see value loss spike above 0.2, something is still wrong (check environment or other configs).

## Technical Details

### Gradient Clipping (max_grad_norm)
Clips gradient norm to 0.5 before parameter update:
```python
if grad_norm > 0.5:
    grads = grads * (0.5 / grad_norm)
```
This prevents single bad batch from making huge parameter changes.

### Advantage Normalization (use_adv_normalize)
Normalizes advantages to zero mean, unit variance:
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```
This makes policy updates more consistent regardless of value function errors.

### Lower Learning Rate
Reduces parameter update size:
- Original: θ_new = θ_old - 5e-3 * grad
- Safe: θ_new = θ_old - 3e-4 * grad

Smaller steps = less likely to overshoot optimal parameters.

## Troubleshooting

### Q: Training is slower with ppo_safe.yaml
**A:** This is expected. Lower learning rate means slower initial convergence, but you'll reach higher final performance without crashes.

### Q: Can I start with ppo.yaml and switch to ppo_safe.yaml later?
**A:** Yes! If your training is stable at 50M steps, you can continue with ppo.yaml. If you see value loss increasing, switch to ppo_safe.yaml and resume from the last stable checkpoint.

### Q: Should I always use ppo_safe.yaml?
**A:** For production training runs to 180M steps, yes. For quick experiments, ppo.yaml is fine.

### Q: What if I still get NaN crashes?
**A:** Check:
1. Is `max_grad_norm: 0.5` actually in the config?
2. Is OpenRL version compatible? (Try OpenRL >= 0.2.0)
3. Are your reward scales reasonable? (Check task config)
4. GPU memory issues? (Use `helpers/gpu_monitor.py`)

## References

- **Crash Analysis:** See `helpers/claude.md` "Known Issues and Solutions" section
- **Original Issue:** Run8 (11-07-17_cuboid) crash at 97.3M steps
- **Research:** [PPO Paper](https://arxiv.org/abs/1707.06347), [Value Function Divergence](https://arxiv.org/abs/1812.02648)

---

**Created:** 2025-11-11
**Status:** Tested configuration changes only (no code modifications)

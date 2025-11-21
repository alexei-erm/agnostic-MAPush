# HAPPO Testing Guide
## How to Test HAPPO Checkpoints

**Date Created**: 2025-11-18
**Status**: ✅ Complete and Tested
**Related Files**: `HARL/docs/HAPPO_INTEGRATION_STATUS.md`

---

## Overview

HAPPO checkpoints have a different structure than MAPPO checkpoints. Instead of a single `module.pt` file, HAPPO saves multiple files per checkpoint:

```
rl_model_{steps}_steps/
├── actor_agent0.pt        # Policy for agent 0
├── actor_agent1.pt        # Policy for agent 1
├── critic_agent.pt        # Centralized critic
└── value_normalizer.pt    # Value normalization parameters
```

---

## Checkpoint Location

HAPPO checkpoints are saved in:
```
results/mapush/go1push_mid/{algo}/{exp_name}/seed-{seed}-{timestamp}/models/
```

Example:
```
results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-15-15-34-13/models/
├── rl_model_2500000_steps/
├── rl_model_5000000_steps/
├── rl_model_7500000_steps/
├── rl_model_10000000_steps/
...
└── rl_model_100000000_steps/
```

Each directory contains the 4 checkpoint files mentioned above.

---

## Testing Methods

### Method 1: Using task/cuboid/train.sh (Recommended)

The `train.sh` script now automatically handles HAPPO testing:

```bash
# Make sure algo="happo" is set in task/cuboid/train.sh
source task/cuboid/train.sh True
```

**What it does:**
1. Automatically finds the latest HAPPO training run
2. Lists all available checkpoints
3. Selects the latest checkpoint (100M steps by default)
4. Runs rendering with 10 episodes

**To test a specific checkpoint:**

Edit `task/cuboid/train.sh` around line 113 and modify:
```bash
# Default: use the latest checkpoint
checkpoint_dir=$(ls -dt "$latest_seed_dir/models"/rl_model_*_steps 2>/dev/null | head -n 1)
```

To select a specific checkpoint:
```bash
# Use 50M steps checkpoint
checkpoint_dir="$latest_seed_dir/models/rl_model_50000000_steps"
```

### Method 2: Direct Python Script

Use the dedicated rendering script directly:

```bash
/home/gvlab/miniconda3/envs/mapush/bin/python HARL/examples/render_mapush.py \
    --algo happo \
    --model_dir results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-15-15-34-13/models/rl_model_100000000_steps \
    --render_episodes 10 \
    --num_envs 1 \
    --seed 5
```

**Parameters:**
- `--algo`: Algorithm (happo, hatrpo, haa2c)
- `--model_dir`: **Full path to checkpoint directory** (containing actor_agent*.pt files)
- `--render_episodes`: Number of episodes to render
- `--num_envs`: Number of parallel environments (usually 1 for visualization)
- `--seed`: Random seed for reproducibility
- `--record_video`: (Optional) Record video to docs/video/
- `--headless`: (Optional) Run without viewer
- `--task`: Task name (default: go1push_mid)

---

## Comparison: MAPPO vs HAPPO Testing

| Feature | MAPPO (OpenRL) | HAPPO (HARL) |
|---------|----------------|--------------|
| **Checkpoint files** | 1 file (`module.pt`) | 4 files (actor0, actor1, critic, normalizer) |
| **Checkpoint location** | `results/{date}_{object}/checkpoints/` | `results/mapush/go1push_mid/{algo}/{exp_name}/` |
| **Testing script** | `openrl_ws/test.py` | `HARL/examples/render_mapush.py` |
| **Test command** | `source task/cuboid/train.sh True` | `source task/cuboid/train.sh True` (same!) |
| **Calculator mode** | ✅ Available | ⏳ To be implemented |

---

## Video Recording

To record videos of HAPPO episodes:

### Method 1: Via train.sh

Edit `task/cuboid/train.sh` line 136-137:
```bash
--task go1push_mid \
--record_video  # Uncomment this line
# --headless    # Comment this out to see viewer
```

### Method 2: Direct script

```bash
/home/gvlab/miniconda3/envs/mapush/bin/python HARL/examples/render_mapush.py \
    --algo happo \
    --model_dir results/mapush/go1push_mid/happo/cuboid/.../rl_model_100000000_steps \
    --render_episodes 3 \
    --record_video \
    --seed 5
```

**Output location:** `docs/video/` (similar to MAPPO)

---

## Success Rate Calculation (TODO)

Currently, HAPPO doesn't have a "calculator mode" like MAPPO. This needs to be implemented.

**MAPPO calculator mode:**
```bash
python ./openrl_ws/test.py \
    --test_mode calculator \
    --num_envs 300 \
    --checkpoint path/to/module.pt
```

**HAPPO equivalent (to be implemented):**
```bash
# TODO: Create success rate evaluation script
python HARL/examples/eval_mapush.py \
    --algo happo \
    --model_dir path/to/rl_model_*_steps/ \
    --num_envs 300
```

---

## Testing Multiple Checkpoints

To test multiple checkpoints and compare performance:

```bash
# List all available checkpoints
ls -d results/mapush/go1push_mid/happo/cuboid/seed-*/models/rl_model_*_steps/

# Test each one manually by editing train.sh
# Or create a loop script:

for checkpoint in results/mapush/go1push_mid/happo/cuboid/seed-00001-*/models/rl_model_*_steps; do
    echo "Testing $checkpoint"
    /home/gvlab/miniconda3/envs/mapush/bin/python HARL/examples/render_mapush.py \
        --algo happo \
        --model_dir "$checkpoint" \
        --render_episodes 5 \
        --headless \
        --seed 5
done
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'harl'"

**Solution:** Use the full python path:
```bash
/home/gvlab/miniconda3/envs/mapush/bin/python HARL/examples/render_mapush.py ...
```

Or activate conda environment first:
```bash
conda activate mapush
python HARL/examples/render_mapush.py ...
```

### Issue: "Missing checkpoint files: ['actor_agent0.pt', ...]"

**Solution:** Make sure you're pointing to the checkpoint **directory**, not the parent folder:
- ✅ Correct: `models/rl_model_100000000_steps`
- ❌ Wrong: `models/`

### Issue: "Isaac Gym GLFW initialization failed"

**Solution:** Same as MAPPO - need to run locally or via VNC. SSH doesn't work for rendering.

### Issue: Checkpoint directory doesn't exist

**Solution:** Check training results location:
```bash
ls -la results/mapush/go1push_mid/happo/
```

Training might still be in progress, or saved under a different exp_name.

---

## Example Workflows

### Workflow 1: Quick visual test of latest checkpoint

```bash
# 1. Set algo to happo in train.sh
vim task/cuboid/train.sh  # Set algo="happo"

# 2. Run test
source task/cuboid/train.sh True
```

### Workflow 2: Record video of specific checkpoint

```bash
/home/gvlab/miniconda3/envs/mapush/bin/python HARL/examples/render_mapush.py \
    --algo happo \
    --model_dir results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-15-15-34-13/models/rl_model_50000000_steps \
    --render_episodes 5 \
    --record_video \
    --seed 5
```

### Workflow 3: Compare different training stages

```bash
# Test early training (10M steps)
python HARL/examples/render_mapush.py --algo happo \
    --model_dir .../models/rl_model_10000000_steps --render_episodes 3

# Test mid training (50M steps)
python HARL/examples/render_mapush.py --algo happo \
    --model_dir .../models/rl_model_50000000_steps --render_episodes 3

# Test final (100M steps)
python HARL/examples/render_mapush.py --algo happo \
    --model_dir .../models/rl_model_100000000_steps --render_episodes 3
```

---

## Files Reference

### Created for HAPPO Testing:
1. **`HARL/examples/render_mapush.py`**
   - Main rendering script for HAPPO models
   - Handles checkpoint loading and episode execution
   - ~150 lines

2. **`task/cuboid/train.sh`** (modified)
   - Lines 81-150: Updated test mode to support HAPPO
   - Automatically detects algorithm and routes to correct testing method
   - Finds latest checkpoint and lists available options

3. **`helpers_claude/HAPPO_TESTING_GUIDE.md`** (this file)
   - Complete documentation for testing HAPPO models

### Related HARL Files:
- **`HARL/harl/runners/on_policy_base_runner.py`**
  - Contains `restore()` method (lines 773-794)
  - Contains `render()` method (lines 594-700)
- **`HARL/harl/configs/algos_cfgs/happo.yaml`**
  - `render` section defines rendering parameters

---

## Next Steps (TODO)

1. **✅ Rendering/testing**: Complete and working
2. **⏳ Success rate calculator**: Need to implement
3. **⏳ Batch checkpoint testing**: Script to test all checkpoints automatically
4. **⏳ Performance comparison**: Script to compare HAPPO vs MAPPO metrics

---

## Summary

**How to test HAPPO models (Quick Reference):**

```bash
# Quick test (recommended)
source task/cuboid/train.sh True

# Custom test
/home/gvlab/miniconda3/envs/mapush/bin/python HARL/examples/render_mapush.py \
    --algo happo \
    --model_dir results/mapush/go1push_mid/happo/cuboid/seed-*/models/rl_model_*_steps \
    --render_episodes 10
```

**Key differences from MAPPO:**
- 4 checkpoint files instead of 1
- Different directory structure
- Use `render_mapush.py` instead of `openrl_ws/test.py`
- Calculator mode not yet implemented

---

**Last Updated**: 2025-11-18
**Status**: ✅ Fully functional for rendering/visualization
**Author**: Claude Code

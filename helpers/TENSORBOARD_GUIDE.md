# TensorBoard Guide - MAPush Remote Monitoring

## Overview

This guide covers how to use TensorBoard to monitor MAPush training in real-time when working remotely from a Mac laptop connected via Tailscale SSH to the Ubuntu training server.

**Your Setup**:
- **Training Server**: Ubuntu desktop (GPU workstation)
- **Local Machine**: Mac laptop
- **Connection**: Tailscale SSH
- **Goal**: Monitor training metrics in real-time from your Mac browser

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Enabling TensorBoard Logging](#enabling-tensorboard-logging)
3. [Remote Access via SSH Tunneling](#remote-access-via-ssh-tunneling)
4. [Understanding the Metrics](#understanding-the-metrics)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Step 1: Start Training with TensorBoard (on Ubuntu Server)

```bash
# SSH into your Ubuntu server
ssh <your-tailscale-hostname-or-ip>

# Activate environment
conda activate mapush
cd /home/gvlab/MAPush

# Start training with TensorBoard enabled
python ./openrl_ws/train.py \
    --algo ppo \
    --task go1push_mid \
    --num_envs 200 \
    --train_timesteps 100000000 \
    --exp_name cuboid \
    --use_tensorboard \
    --headless
```

### Step 2: Start TensorBoard Server (on Ubuntu Server)

**Option A: In a new SSH session**:
```bash
# Open a new terminal on your Mac
ssh <your-tailscale-hostname-or-ip>

# Activate environment
conda activate mapush
cd /home/gvlab/MAPush

# Start TensorBoard (monitoring latest run)
tensorboard --logdir=./log --port=6006 --bind_all
```

**Option B: Use screen/tmux** (recommended):
```bash
# In your existing SSH session
screen -S tensorboard

# Start TensorBoard
conda activate mapush
cd /home/gvlab/MAPush
tensorboard --logdir=./log --port=6006 --bind_all

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r tensorboard
```

### Step 3: Create SSH Tunnel (on Mac Laptop)

**Open a new terminal on your Mac**:

```bash
# Create SSH tunnel to forward port 6006
ssh -L 6006:localhost:6006 <your-tailscale-hostname-or-ip>

# Keep this terminal window open while you use TensorBoard
```

**What this does**:
- `-L 6006:localhost:6006` forwards local port 6006 to remote port 6006
- TensorBoard running on Ubuntu becomes accessible at `localhost:6006` on your Mac

### Step 4: Access TensorBoard (on Mac Browser)

Open your browser (Chrome/Safari/Firefox) and navigate to:

```
http://localhost:6006
```

You should see the TensorBoard interface with real-time training metrics!

---

## Enabling TensorBoard Logging

### Command-Line Flag

Add `--use_tensorboard` to any training command:

```bash
python ./openrl_ws/train.py \
    --algo ppo \
    --task go1push_mid \
    --num_envs 200 \
    --train_timesteps 100000000 \
    --exp_name my_experiment \
    --use_tensorboard \
    --headless
```

### Log Directory Structure

TensorBoard logs are saved in **two locations**:

**1. Active Training Logs** (while training is running):
```
./log/MQE/<task>/<timestamp>/
```

Example:
```
/home/gvlab/MAPush/log/MQE/go1push_mid/2024-10-15_23-15-30/
```

**2. Archived Logs** (after training completes):
```
./results/<timestamp>_<exp_name>/logs/
```

Example:
```
/home/gvlab/MAPush/results/10-15-23_cuboid/logs/
├── actor_grad_norm/
├── approach_to_box_reward/
├── average_step_reward/
├── collision_punishment/
├── critic_grad_norm/
├── distance_to_target_reward/
├── ocb_reward/
├── policy_loss/
├── push_reward/
├── reach_target_reward/
├── value_loss/
└── events.out.tfevents.*
```

---

## Remote Access via SSH Tunneling

### Method 1: Simple Port Forwarding (Recommended)

**On Mac Terminal 1** (SSH + Training):
```bash
ssh <your-ubuntu-host>
conda activate mapush
cd /home/gvlab/MAPush
python ./openrl_ws/train.py --use_tensorboard ...
```

**On Mac Terminal 2** (TensorBoard Server):
```bash
ssh <your-ubuntu-host>
conda activate mapush
tensorboard --logdir=/home/gvlab/MAPush/log --port=6006 --bind_all
```

**On Mac Terminal 3** (SSH Tunnel):
```bash
ssh -L 6006:localhost:6006 <your-ubuntu-host>
# Keep this running
```

**On Mac Browser**:
```
http://localhost:6006
```

### Method 2: Single SSH Command with Port Forwarding

**On Mac**:
```bash
# Connect with port forwarding in one command
ssh -L 6006:localhost:6006 <your-ubuntu-host>

# Then start TensorBoard on the remote server
conda activate mapush
tensorboard --logdir=/home/gvlab/MAPush/log --port=6006 --bind_all
```

**On Mac Browser**:
```
http://localhost:6006
```

### Method 3: Background SSH Tunnel

**On Mac**:
```bash
# Create persistent background tunnel
ssh -f -N -L 6006:localhost:6006 <your-ubuntu-host>

# -f: Background process
# -N: Don't execute remote command
# -L: Port forwarding

# TensorBoard is now accessible at localhost:6006
# Kill the tunnel later with: pkill -f "ssh -f -N -L 6006"
```

### Tailscale-Specific Setup

**Find your Tailscale hostname**:
```bash
# On Ubuntu server
tailscale status

# Output shows something like:
# 100.x.x.x   my-ubuntu-machine   user@      linux   -
```

**Connect using Tailscale name**:
```bash
# On Mac
ssh -L 6006:localhost:6006 my-ubuntu-machine

# Or use Tailscale IP
ssh -L 6006:localhost:6006 100.x.x.x
```

### Using Different Ports

If port 6006 is already in use:

**On Ubuntu** (TensorBoard):
```bash
tensorboard --logdir=./log --port=6007 --bind_all
```

**On Mac** (SSH tunnel):
```bash
ssh -L 6007:localhost:6007 <your-ubuntu-host>
```

**On Mac Browser**:
```
http://localhost:6007
```

---

## Understanding the Metrics

### Training Progress Metrics

#### **average_step_reward**
- **What**: Mean reward per environment step
- **Good trend**: Steadily increasing over training
- **Target**: Should increase from ~0 to positive values
- **Usage**: Overall training progress indicator

#### **policy_loss**
- **What**: PPO policy (actor) loss
- **Good trend**: Decreases initially, then stabilizes
- **Target**: Low and stable (typically < 0.5)
- **Usage**: Indicates policy learning stability

#### **value_loss**
- **What**: Value function (critic) loss
- **Good trend**: Decreases over time
- **Target**: Low values indicate good value estimation
- **Usage**: Monitors critic network learning

### Reward Components

#### **distance_to_target_reward**
- **What**: Reward for reducing distance to goal
- **Scale**: 0.00325 per meter reduced
- **Good trend**: Increases as policy learns to push toward target
- **File**: `task/cuboid/config.py` → `rewards.scales.target_reward_scale`

#### **approach_to_box_reward**
- **What**: Reward for robots approaching the box
- **Scale**: 0.00075
- **Good trend**: High initially, may decrease as pushing improves
- **Usage**: Ensures robots engage with the box

#### **push_reward**
- **What**: Reward for actively pushing the box
- **Scale**: 0.0015
- **Good trend**: Increases over training
- **Usage**: Encourages forward pushing motion

#### **ocb_reward** (Optimal Collaborative Behavior)
- **What**: Reward for coordinated multi-agent pushing
- **Scale**: 0.004
- **Good trend**: Increases as collaboration improves
- **Usage**: **Critical metric** for multi-agent coordination
- **Target**: High values (>0.003) indicate good collaboration

#### **reach_target_reward**
- **What**: Bonus reward for successfully reaching target
- **Scale**: 10.0 (large bonus)
- **Good trend**: Increases as success rate improves
- **Usage**: Tracks successful episodes

### Penalties

#### **collision_punishment**
- **What**: Penalty for robot-robot collisions
- **Scale**: -0.0025 per collision
- **Good trend**: Should decrease (less negative) over training
- **Target**: Near zero (no collisions)
- **Usage**: Monitors safety and coordination

#### **exception_punishment**
- **What**: Penalty for early termination (falls, excessive tilt)
- **Scale**: -5.0 per termination
- **Good trend**: Should decrease over training
- **Usage**: Tracks policy stability

### Gradient Metrics

#### **actor_grad_norm**
- **What**: Gradient norm of policy network
- **Good trend**: Stable, not exploding or vanishing
- **Warning**: If >10, gradients may be unstable
- **Usage**: Diagnose training instability

#### **critic_grad_norm**
- **What**: Gradient norm of value network
- **Good trend**: Stable, moderate values
- **Usage**: Monitor critic learning stability

### Policy Metrics

#### **dist_entropy**
- **What**: Action distribution entropy (exploration)
- **Good trend**: High initially, decreases over time
- **Target**: Gradual decrease as policy becomes more deterministic
- **Usage**: Monitors exploration vs. exploitation balance

#### **ratio**
- **What**: PPO clipping ratio
- **Good trend**: Should stay near 1.0
- **Target**: Within [0.8, 1.2] for stable PPO training
- **Usage**: Monitors policy update magnitude

---

## Advanced Usage

### Monitoring Multiple Experiments

**Compare different training runs**:

```bash
# On Ubuntu
tensorboard --logdir_spec=\
cuboid_run1:./results/10-15-23_cuboid/logs,\
cuboid_run2:./results/10-28-18_cuboid/logs,\
tblock:./results/10-15-23_Tblock/logs \
--port=6006 --bind_all
```

**On Mac browser**, you'll see all runs overlayed for comparison.

### Monitoring Specific Metrics

**Only reward metrics**:
```bash
tensorboard --logdir=./results/10-15-23_cuboid/logs \
    --path_prefix=/reward \
    --port=6006 --bind_all
```

### Real-Time Training Monitoring

**Setup auto-refresh** (TensorBoard updates automatically, but you can force faster updates):

```bash
# On Ubuntu
tensorboard --logdir=./log --port=6006 --bind_all --reload_interval=30

# Reloads every 30 seconds (default is 60)
```

### Using tmux for Persistent Sessions

**Recommended setup** for long training runs:

```bash
# On Ubuntu (via SSH from Mac)
tmux new -s training

# Split screen horizontally
Ctrl+B, then "

# Top pane: Training
conda activate mapush
cd /home/gvlab/MAPush
python ./openrl_ws/train.py --use_tensorboard ...

# Bottom pane: TensorBoard
Ctrl+B, then Down Arrow
conda activate mapush
tensorboard --logdir=./log --port=6006 --bind_all

# Detach from tmux
Ctrl+B, then D

# Reattach later
tmux attach -t training
```

**On Mac**, create persistent SSH tunnel:
```bash
ssh -L 6006:localhost:6006 <ubuntu-host>
```

### Exporting TensorBoard Data

**Export scalar data to CSV** (on Ubuntu):

```bash
conda activate mapush

# Install tensorboard plugin
pip install tensorboard-plugin-profile

# Export data
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

event_acc = EventAccumulator('./results/10-15-23_cuboid/logs')
event_acc.Reload()

# Get scalar tags
tags = event_acc.Tags()['scalars']

# Export each metric
for tag in tags:
    events = event_acc.Scalars(tag)
    df = pd.DataFrame(events)
    df.to_csv(f'{tag.replace(\"/\", \"_\")}.csv', index=False)
    print(f'Exported {tag}')
"
```

### Monitoring from Multiple Locations

**If you want to access TensorBoard from multiple devices**:

**On Ubuntu**, start TensorBoard with `--bind_all`:
```bash
tensorboard --logdir=./log --port=6006 --bind_all
```

**On Mac 1**:
```bash
ssh -L 6006:localhost:6006 <ubuntu-host>
# Access at http://localhost:6006
```

**On Mac 2** (different port to avoid conflict):
```bash
ssh -L 6007:localhost:6006 <ubuntu-host>
# Access at http://localhost:6007
```

---

## Troubleshooting

### Issue 1: "Address already in use" (Port 6006)

**Error**:
```
TensorBoard could not bind to port 6006, it was already in use
```

**Solution 1** - Use different port:
```bash
# On Ubuntu
tensorboard --logdir=./log --port=6007 --bind_all

# On Mac
ssh -L 6007:localhost:6007 <ubuntu-host>
```

**Solution 2** - Kill existing TensorBoard:
```bash
# On Ubuntu
pkill -f tensorboard

# Or find and kill specific process
lsof -ti:6006 | xargs kill -9
```

### Issue 2: SSH Tunnel Connection Refused

**Error** on Mac:
```
channel 2: open failed: connect failed: Connection refused
```

**Solution**: Make sure TensorBoard is running on Ubuntu:
```bash
# SSH into Ubuntu
ssh <ubuntu-host>

# Check if TensorBoard is running
ps aux | grep tensorboard

# If not running, start it
conda activate mapush
tensorboard --logdir=/home/gvlab/MAPush/log --port=6006 --bind_all
```

### Issue 3: TensorBoard Shows No Data

**Possible causes**:

1. **Training hasn't started yet**:
   - Wait for first logging interval (usually 20-60 seconds)

2. **Wrong log directory**:
   ```bash
   # Check where logs actually are
   find /home/gvlab/MAPush -name "events.out.tfevents.*" -type f

   # Point TensorBoard to correct location
   tensorboard --logdir=<correct-path> --port=6006 --bind_all
   ```

3. **TensorBoard not enabled**:
   - Make sure you used `--use_tensorboard` flag in training command

4. **Logs moved to results directory**:
   ```bash
   # If training finished, logs are in results/
   tensorboard --logdir=/home/gvlab/MAPush/results/10-15-23_cuboid/logs --port=6006 --bind_all
   ```

### Issue 4: SSH Tunnel Keeps Disconnecting

**Solution 1** - Add keep-alive to SSH config:

On Mac, edit `~/.ssh/config`:
```
Host <your-ubuntu-host>
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

**Solution 2** - Use autossh (auto-reconnect):
```bash
# Install autossh on Mac
brew install autossh

# Use autossh instead of ssh
autossh -M 0 -L 6006:localhost:6006 <ubuntu-host>
```

### Issue 5: Browser Shows "localhost refused to connect"

**Checklist**:

1. ✅ Is TensorBoard running on Ubuntu?
   ```bash
   ssh <ubuntu-host> "ps aux | grep tensorboard"
   ```

2. ✅ Is SSH tunnel active on Mac?
   ```bash
   ps aux | grep "ssh -L 6006"
   ```

3. ✅ Is port 6006 correct in all places?
   - TensorBoard port on Ubuntu
   - SSH tunnel port mapping
   - Browser URL

4. ✅ Try accessing from Ubuntu directly:
   ```bash
   # SSH into Ubuntu
   curl http://localhost:6006
   # Should return HTML
   ```

### Issue 6: Metrics Not Updating

**Solution**: Force TensorBoard refresh:

```bash
# On Ubuntu, restart TensorBoard with faster reload
pkill -f tensorboard
tensorboard --logdir=./log --port=6006 --bind_all --reload_interval=15
```

**In browser**: Hard refresh with `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R`

### Issue 7: Can't Find Recent Training Logs

**Solution**: Check both log locations:

```bash
# Active training logs
ls -lt /home/gvlab/MAPush/log/MQE/*/

# Completed training logs
ls -lt /home/gvlab/MAPush/results/*/logs/

# Point TensorBoard to most recent
tensorboard --logdir=/home/gvlab/MAPush/log --port=6006 --bind_all
```

---

## Complete Workflow Example

**Scenario**: Start new training and monitor from Mac

### On Mac - Terminal 1 (Training):

```bash
# SSH into Ubuntu
ssh my-ubuntu-machine

# Start training with TensorBoard
conda activate mapush
cd /home/gvlab/MAPush

python ./openrl_ws/train.py \
    --algo ppo \
    --task go1push_mid \
    --num_envs 200 \
    --train_timesteps 100000000 \
    --exp_name my_experiment \
    --use_tensorboard \
    --headless
```

### On Mac - Terminal 2 (TensorBoard):

```bash
# SSH into Ubuntu
ssh my-ubuntu-machine

# Start TensorBoard server
conda activate mapush
cd /home/gvlab/MAPush
tensorboard --logdir=./log --port=6006 --bind_all
```

### On Mac - Terminal 3 (SSH Tunnel):

```bash
# Create port forwarding tunnel
ssh -L 6006:localhost:6006 my-ubuntu-machine

# Keep this terminal open
```

### On Mac - Browser:

```
Open: http://localhost:6006
```

**You should see**:
- Real-time training metrics updating
- Reward components graphs
- Loss curves
- Gradient norms

---

## Best Practices

### 1. Use Screen/Tmux for Persistence

```bash
# On Ubuntu
screen -S training
python ./openrl_ws/train.py --use_tensorboard ...
# Ctrl+A, D to detach

screen -S tensorboard
tensorboard --logdir=./log --port=6006 --bind_all
# Ctrl+A, D to detach

# Detach and close Mac laptop - training continues!
# Reconnect later: screen -r training
```

### 2. Monitor Key Metrics

Focus on these during training:
- ✅ **average_step_reward**: Should increase steadily
- ✅ **ocb_reward**: Should increase (collaboration improving)
- ✅ **collision_punishment**: Should approach zero
- ✅ **reach_target_reward**: Should increase (more successes)

### 3. Save TensorBoard Snapshots

Take screenshots of important milestone metrics for your research notes.

### 4. Compare Experiments

Use multiple log directories to compare different hyperparameters:
```bash
tensorboard --logdir_spec=\
exp1:./results/10-15-23_cuboid/logs,\
exp2:./results/10-28-18_cuboid/logs \
--port=6006 --bind_all
```

### 5. Archive Logs

After training completes, logs are automatically saved to:
```
./results/<timestamp>_<exp_name>/logs/
```

Keep these for future reference and paper figures!

---

## Alternative: WandB (Weights & Biases)

The repo also supports **Weights & Biases** for cloud-based monitoring:

```bash
# Train with WandB instead of TensorBoard
python ./openrl_ws/train.py \
    --use_wandb \
    --wandb_entity <your-entity> \
    ...
```

**Advantages**:
- No SSH tunneling needed
- Access from anywhere via web
- Better experiment comparison tools

**Disadvantages**:
- Requires account setup
- Uploads data to cloud (slower on some networks)

---

## Summary

### Quick Command Reference

**Start training with TensorBoard**:
```bash
python ./openrl_ws/train.py --use_tensorboard --headless ...
```

**Start TensorBoard server**:
```bash
tensorboard --logdir=./log --port=6006 --bind_all
```

**Create SSH tunnel** (Mac):
```bash
ssh -L 6006:localhost:6006 <ubuntu-host>
```

**Access in browser** (Mac):
```
http://localhost:6006
```

**Compare multiple runs**:
```bash
tensorboard --logdir_spec=run1:./results/run1/logs,run2:./results/run2/logs --port=6006 --bind_all
```

---

**Happy Monitoring!**

For related documentation:
- `TESTING_GUIDE.md` - Testing trained policies
- `CHECKPOINT_SYSTEM.md` - Checkpoint management
- `README.md` - Installation and basic usage

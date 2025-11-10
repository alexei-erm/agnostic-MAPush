# Multi-Episode Video Recording Implementation

## Overview

This document describes the implementation of multi-episode video recording for Isaac Gym environments with controllable seed values. The solution allows recording multiple consecutive episodes back-to-back in a single MP4 file without relying on command-line arguments.

## Problem Statement

The original implementation had several limitations:
1. Could only record one episode at a time
2. Command-line flags for seed control (`--seed`) were not properly passed through the wrapper layers
3. Recording would stop after the first episode completed
4. No way to specify number of episodes to record via arguments

## Solution Architecture

### 1. Hardcoded Configuration

Instead of relying on argument parsing which had compatibility issues, we implemented hardcoded constants at the top of `openrl_ws/test.py`:

```python
# ============== HARDCODED CONFIG ==============
SEED = 5  # Set your desired seed here
NUM_EPISODES = 3  # Number of consecutive episodes to record in one video
# ==============================================
```

**Benefits:**
- Simple and reliable - no argument parsing issues
- Easy to modify before each test run
- Works regardless of wrapper complexity

### 2. Seed Control Implementation

The seed is injected directly into the args object before environment creation:

```python
args = get_args()
args.seed = SEED
env, _ = make_env(args, custom_cfg(args))
```

This ensures that when `make_mqe_env()` internally calls `set_seed(args.seed)`, it uses our hardcoded value.

**Why this approach:**
- Avoids conflicts with existing seed-setting mechanisms in `mqe/utils/helpers.py:255`
- Works through all wrapper layers
- Maintains compatibility with existing code

### 3. Multi-Episode Recording via Monkey-Patching

The core challenge was that the recording system (`store_recording()` and `_render_headless()`) was designed for single episodes. We solved this by patching these methods at runtime.

#### 3.1 Environment Unwrapping

First, we unwrap all wrapper layers to reach the actual `Go1Object` environment where recording happens:

```python
actual_env = env
while hasattr(actual_env, 'env'):
    actual_env = actual_env.env
```

**Wrapper hierarchy discovered:**
```
MATWrapper → mqe_openrl_wrapper → Go1PushMidWrapper → Go1Object (actual env)
```

#### 3.2 Patching store_recording()

We replace the `store_recording()` method to accumulate frames across multiple episodes:

```python
def multi_episode_store_recording(env_ids):
    if actual_env.cfg.env.record_video and 0 in env_ids:
        if len(actual_env.video_frames) > 0:
            actual_env._recording_episodes_count += 1

            # Accumulate all frames
            actual_env._all_video_frames.extend(actual_env.video_frames)
            actual_env.video_frames = []

            # Only finalize when all episodes are done
            if actual_env._recording_episodes_count >= actual_env._recording_episodes_target:
                actual_env.complete_video_frames = actual_env._all_video_frames[:]
                actual_env.record_now = False
```

**Key insight:** The original implementation would clear `video_frames` and set `complete_video_frames` after EACH episode. Our patch accumulates frames from all episodes before finalizing.

#### 3.3 Patching _render_headless()

The original `_render_headless()` had a condition that prevented recording after the first episode:

```python
# ORIGINAL (problematic):
if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
```

We removed the `len(self.complete_video_frames) == 0` check:

```python
def patched_render_headless():
    if actual_env.record_now and actual_env.complete_video_frames is not None:
        actual_env.rendering_camera.set_position(actual_env.cfg.viewer.pos, actual_env.cfg.viewer.lookat)
        actual_env.video_frame = actual_env.rendering_camera.get_observation()
        actual_env.video_frames.append(actual_env.video_frame)
```

**Why:** This allows continuous frame capture across all episodes.

#### 3.4 Bypassing Gym's Private Attribute Protection

Gym wrappers block access to attributes starting with `_`. We use `object.__setattr__()` to bypass this:

```python
object.__setattr__(actual_env, '_render_headless', patched_render_headless)
```

### 4. Video Output

Videos are saved with informative filenames:
```python
filename = f"test_seed{SEED}_{NUM_EPISODES}eps.mp4"
# Example: test_seed5_3eps.mp4
```

Saved to: `docs/video/` directory

## Files Modified

### 1. `openrl_ws/test.py`

**Changes:**
- Added hardcoded `SEED` and `NUM_EPISODES` constants (lines 20-23)
- Injected seed into args (lines 113-114)
- Implemented environment unwrapping (lines 168-171)
- Added multi-episode recording monkey-patches (lines 173-208)
- Modified video output filename (line 230)

### 2. `mqe/envs/base/legged_robot.py`

**Changes made (optional, for future improvements):**
- Modified `_render_headless()` to remove empty check (line 1182)
- Updated `start_recording()` to accept `num_episodes` parameter (line 1192)
- Enhanced `store_recording()` with episode counting (lines 1214-1224)
- Added default episode tracking attributes (lines 74-75)

**Note:** The monkey-patching in test.py makes these base file changes optional, but they're included for cleaner future integration.

## Usage

### Basic Usage

1. **Edit configuration** in `openrl_ws/test.py`:
   ```python
   SEED = 42  # Your desired seed
   NUM_EPISODES = 5  # Number of episodes to record
   ```

2. **Run the test:**
   ```bash
   source ./results/11-07-17_cuboid/task/train.sh True
   ```

3. **Check output:**
   - Video saved to: `docs/video/test_seed42_5eps.mp4`
   - Contains all 5 episodes back-to-back

### Expected Output

```
Using seed: 42
Recording 5 consecutive episodes...
Stored episode 1/5 (257 frames)
Episode 1/5: success
Stored episode 2/5 (311 frames)
Episode 2/5: success
...
All 5 episodes recorded! Total frames: 1234
Video shape: (1234, 4, 1080, 1080)
video has been created!
```

## Remote/Headless Execution Challenges

### The Problem

When running via SSH (headless mode), Isaac Gym's GLFW backend fails to initialize:
```
[Error] [carb.windowing-glfw.plugin] GLFW initialization failed.
```

This causes the rendering camera to capture **static/blank frames** - all frames have identical mean values (e.g., 212.365...), indicating the graphics pipeline isn't updating.

### Why It Happens

- Isaac Gym's rendering camera requires proper graphics context initialization
- GLFW cannot initialize without a display (X11/Wayland)
- Over SSH, even with `DISPLAY=:0` set, GLFW fails because there's no actual window manager available to the SSH session

### Working Solutions

#### ✅ Option 1: Run Locally (Recommended)
Open terminal directly on the desktop machine (not via SSH):
```bash
cd ~/agnostic-MAPush
source ./results/11-07-17_cuboid/task/train.sh True
```

Then copy video to your remote machine:
```bash
scp gvlab@desktop:~/agnostic-MAPush/docs/video/test_seed*.mp4 ./
```

#### ✅ Option 2: VNC/Remote Desktop

Install and use VNC with GPU support:

**TurboVNC + VirtualGL (Recommended):**
```bash
# On server
sudo apt-get install -y turbovnc virtualgl
/opt/TurboVNC/bin/vncserver -geometry 1920x1080 -depth 24

# From client
# Connect with TurboVNC Viewer to server:5901
# Run test in VNC terminal
```

**TigerVNC (Alternative):**
```bash
# On server
sudo apt-get install -y tigervnc-standalone-server tigervnc-xorg-extension
vncserver :1 -geometry 1920x1080 -depth 24

# From client
# Connect with VNC client to server:5901
```

**Verify GPU access in VNC session:**
```bash
glxinfo | grep "direct rendering"
# Should show: "direct rendering: Yes"
```

#### ❌ What Doesn't Work

- **Plain SSH with DISPLAY forwarding** - GLFW still fails
- **xvfb-run** - Lacks GLX extension needed by Isaac Gym
- **EGL offscreen rendering** - Isaac Gym doesn't support this backend
- **X11 forwarding** - Too slow and GLFW still fails

### Script Configuration

The `train.sh` script attempts to use the desktop's display:

```bash
# Attempt to access local X display for rendering
if [ -z "$DISPLAY" ]; then
    for disp in :0 :1; do
        if xdpyinfo -display $disp >/dev/null 2>&1; then
            export DISPLAY=$disp
            break
        fi
    done
fi
```

**Required setup on desktop:**
```bash
xhost +local:
```

This allows local processes (including SSH sessions) to access the display.

## Technical Details

### Frame Capture Flow

1. **Environment step** → triggers `post_physics_step()`
2. **Reset on done** → calls `store_recording(env_ids)`
3. **Our patched method** → accumulates frames instead of finalizing
4. **After N episodes** → moves all frames to `complete_video_frames`
5. **Test loop retrieves** → `get_complete_frames()` returns accumulated frames
6. **Video saved** → as single MP4 file

### Memory Considerations

- Each frame: ~4.5 MB (1080x1080x4 channels at uint8)
- 3 episodes × ~250 frames/episode = ~750 frames
- Total memory: ~3.4 GB for frames buffer

For longer recordings, consider:
- Reducing resolution in config: `env.recording_width_px/recording_height_px`
- Recording fewer episodes
- Writing frames to disk incrementally (requires deeper modification)

## Troubleshooting

### Issue: Video is 0 seconds / only 1 frame

**Symptoms:**
```
Video shape: (1, 4, 1080, 1080)
```

**Cause:** Monkey-patching failed, original methods still in use

**Solution:**
1. Clear Python cache: `find . -name "*.pyc" -delete`
2. Check environment unwrapping prints correct type: `Go1Object`
3. Verify patches applied: Look for "Stored episode X/N" messages

### Issue: All frames identical (blank video)

**Symptoms:**
```
Video mean: 212.3651225994513  # Always same value
```

**Cause:** Running headless without proper display access

**Solution:** Use VNC or run locally on desktop (see "Remote/Headless Execution" section)

### Issue: Segmentation fault after video creation

**Status:** Known issue, doesn't affect video output

**Cause:** Isaac Gym cleanup issue when exiting

**Workaround:** Video is already saved before crash, can be safely ignored

## Future Improvements

1. **Add command-line arguments** (if wrapper issues resolved):
   ```python
   parser.add_argument('--num_episodes', type=int, default=1)
   parser.add_argument('--seed', type=int, default=0)
   ```

2. **Progress bar** for long recordings:
   ```python
   from tqdm import tqdm
   with tqdm(total=NUM_EPISODES) as pbar:
       # update on each episode
   ```

3. **Incremental disk writing** for very long recordings to reduce memory usage

4. **EGL backend support** for proper headless rendering (requires Isaac Gym update)

## Summary

This implementation provides:
- ✅ Hardcoded seed control (no argument parsing issues)
- ✅ Multi-episode recording in single video file
- ✅ Configurable number of episodes
- ✅ Works reliably when run locally
- ✅ Clean output with episode progress tracking

**Limitations:**
- Must run locally or via VNC (not plain SSH)
- Memory usage scales with number of episodes
- Requires manual editing of constants (not command-line arguments)

**Credits:** Implementation developed through collaborative debugging to work around Isaac Gym's rendering and wrapper layer limitations.

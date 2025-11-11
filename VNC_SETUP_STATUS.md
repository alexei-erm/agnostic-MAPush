# VNC Setup Status and Troubleshooting Log

**Date:** 2025-11-10
**Goal:** Enable remote video recording for Isaac Gym via VNC connection from Mac to Linux desktop over Tailscale
**Current Status:** ❌ Not working - connection refused

---

## Background Context

### The Core Problem
Isaac Gym requires **GLX (OpenGL)** support to render frames for video recording. When running via plain SSH:
- GLFW fails to initialize: `[Error] [carb.windowing-glfw.plugin] GLFW initialization failed`
- Rendering camera captures **static/blank frames** (all frames have identical mean: 212.365...)
- Graphics pipeline doesn't update between frames

### Why VNC is Needed
VNC provides a full desktop environment with proper graphics context, allowing Isaac Gym's rendering camera to work correctly.

### Success Criteria
1. Connect to desktop via VNC from Mac (over Tailscale)
2. Open terminal in VNC session
3. Verify GLX works: `glxinfo | grep "direct rendering"` shows "Yes"
4. Run Isaac Gym test: `source ./results/11-07-17_cuboid/task/train.sh True`
5. Video records with actual content (robots, objects, goals visible)

---

## System Information

**Desktop Machine:**
- OS: Ubuntu 22.04
- GPU: NVIDIA (cuda:0)
- Kernel: Linux 6.8.0-87-generic
- User: gvlab
- IP Address: 10.9.73.9 (Tailscale)

**Client Machine:**
- OS: macOS
- VNC Client: RealVNC Viewer
- Connection: Via Tailscale SSH

**Current Desktop State:**
- No user physically logged into GUI
- Desktop session not active (important - affects Service Mode)

---

## Installations Performed

### 1. ✅ xvfb (X Virtual Frame Buffer)
```bash
sudo apt-get install -y xvfb
```
**Status:** Installed
**Purpose:** Create virtual display for headless rendering
**Result:** ❌ Failed - xvfb lacks GLX extension needed by Isaac Gym
**Error:** `Xlib: extension "GLX" missing on display ":99"`

### 2. ✅ RealVNC Server (Version 7.12.0)
```bash
wget https://downloads.realvnc.com/download/file/vnc.files/VNC-Server-7.12.0-Linux-x64.deb
sudo dpkg -i VNC-Server-7.12.0-Linux-x64.deb
sudo apt-get install -f
sudo systemctl enable vncserver-x11-serviced
sudo systemctl start vncserver-x11-serviced
```
**Status:** Installed and running
**Current Service:** `vncserver-x11-serviced` (Service Mode)
**Purpose:** Share existing desktop or create virtual desktop

### 3. ✅ TigerVNC (Already installed)
```bash
# Was already on system
sudo apt-get install -y tigervnc-standalone-server tigervnc-xorg-extension
```
**Status:** Installed
**Purpose:** Alternative VNC server with virtual desktop support

### 4. ❓ VirtualGL (Not installed)
**Status:** Not yet installed
**Purpose:** Provides GPU acceleration for VNC virtual desktops
**Note:** May be needed for GLX support in virtual mode

---

## Configuration Attempts

### Attempt 1: xvfb-run with xhost
**Command:**
```bash
xhost +local:  # On desktop
export DISPLAY=:0  # Via SSH
source ./results/11-07-17_cuboid/task/train.sh True
```
**Result:** ❌ GLFW still failed, xvfb lacks GLX
**Error:** `Xlib: extension "GLX" missing on display ":99"`

### Attempt 2: RealVNC Service Mode (Share Desktop)
**Configuration:**
```bash
sudo systemctl status vncserver-x11-serviced
```
**Service Status:** ✅ Running
**Result:** ❌ Cannot find X server
**Error:** `ConsoleDisplay: Cannot find a running X server on vt2`
**Root Cause:** Service Mode requires someone to be physically logged into the desktop. No one is logged in.

### Attempt 3: TigerVNC Virtual Desktop
**Commands Executed:**
```bash
vncpasswd  # Set password
mkdir -p ~/.vnc
# Created xstartup and config files
vncserver :1 -localhost no -geometry 1920x1080 -depth 24
```
**Result:** ❌ Command hijacked by RealVNC
**Issue:** `vncserver` command calls RealVNC's version, not TigerVNC
**Evidence:** Command output shows RealVNC help instead of starting server
**Process Check:** `ps aux | grep vnc` shows no VNC server running

### Attempt 4: VNC Connection Test
**Client:** RealVNC Viewer on Mac
**Address Tried:** 10.9.73.9:5900
**Result:** ❌ Connection refused
**Reason:** No VNC server actually listening on any port

---

## Current System State

### Active Services
```bash
# RealVNC Service Mode (running but not functional)
sudo systemctl status vncserver-x11-serviced
# Status: active (running)
# Problem: "Cannot find a running X server on vt2"
```

### No VNC Server Actually Running
```bash
ps aux | grep vnc
# Only shows grep process, no actual VNC server
```

### Ports Status (Unknown)
```bash
# Need to check:
sudo netstat -tlnp | grep 590
# Or:
sudo ss -tlnp | grep 590
```

### Firewall Status (Unknown)
```bash
# Need to check:
sudo ufw status
```

---

## Problems Identified

### Problem 1: RealVNC Command Conflict
**Issue:** TigerVNC's `vncserver` command is overridden by RealVNC's version
**Evidence:** Running `vncserver :1` shows RealVNC help/usage
**Impact:** Cannot start TigerVNC server using standard commands

### Problem 2: RealVNC Service Mode Limitation
**Issue:** Service Mode requires physical desktop login
**Current State:** No one logged into desktop GUI
**Impact:** Service cannot attach to non-existent X session

### Problem 3: RealVNC Virtual Mode Not Configured
**Issue:** Virtual Mode (creates its own desktop) not set up
**Commands Not Yet Tried:**
- `vncpasswd -virtual`
- `vncserver-virtual`
- `/etc/vnc/vncservice start vncserver-virtuald`

### Problem 4: No VNC Server Listening
**Issue:** No VNC process running on any port
**Impact:** All connection attempts refused

---

## Next Steps to Try (For Tomorrow)

### Option A: RealVNC Virtual Mode (Proper Setup)
```bash
# 1. Set password for virtual mode
vncpasswd -virtual

# 2. Start virtual mode daemon
sudo /etc/vnc/vncservice start vncserver-virtuald

# 3. Or start directly
vncserver-virtual -display :1 -geometry 1920x1080 -depth 24

# 4. Check status
sudo systemctl status vncserver-virtuald
ps aux | grep vncserver-virtual

# 5. Check logs if failed
journalctl -u vncserver-virtuald -n 50 --no-pager
```

### Option B: Use TigerVNC Directly (Bypass Command Conflict)
```bash
# 1. Stop RealVNC service
sudo systemctl stop vncserver-x11-serviced
sudo systemctl disable vncserver-x11-serviced

# 2. Find TigerVNC binary location
which Xtigervnc
# Or:
find /usr -name "*tigervnc*" 2>/dev/null

# 3. Start TigerVNC directly with full path
/usr/bin/Xtigervnc :1 -geometry 1920x1080 -depth 24 -SecurityTypes None &

# 4. Verify running
ps aux | grep Xtigervnc
sudo netstat -tlnp | grep 5901
```

### Option C: Install and Use TurboVNC (Best for GPU Apps)
```bash
# 1. Download TurboVNC
cd /tmp
wget https://sourceforge.net/projects/turbovnc/files/3.1.1/turbovnc_3.1.1_amd64.deb
sudo dpkg -i turbovnc_3.1.1_amd64.deb

# 2. Install VirtualGL for GPU support
sudo apt-get install -y virtualgl

# 3. Configure VirtualGL
sudo /opt/VirtualGL/bin/vglserver_config
# Answer: 1, n, n

# 4. Add user to vglusers group
sudo usermod -a -G vglusers gvlab
# Must logout/login or: newgrp vglusers

# 5. Start TurboVNC
/opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24

# 6. Connect and test
# In VNC session: vglrun glxgears
```

### Option D: X11 Display Forwarding with GPU (Long Shot)
```bash
# 1. Enable X11 forwarding in SSH
# Edit /etc/ssh/sshd_config:
#   X11Forwarding yes
#   X11DisplayOffset 10
#   X11UseLocalhost no

# 2. Restart SSH
sudo systemctl restart sshd

# 3. Connect with X11 forwarding
ssh -Y gvlab@10.9.73.9

# 4. Export display with GPU
export DISPLAY=:0
xhost +local:

# 5. Test
DISPLAY=:0 glxinfo | grep "direct rendering"
```

---

## Diagnostic Commands Needed

Before trying next solutions, gather information:

```bash
# 1. Check what's listening on VNC ports
sudo netstat -tlnp | grep 590
sudo ss -tlnp | grep 590

# 2. Check firewall
sudo ufw status
# If active, may need: sudo ufw allow 5900:5910/tcp

# 3. Find all VNC-related binaries
which vncserver
which Xtigervnc
which vncserver-virtual
which vncserver-x11
ls -la /usr/bin/*vnc*

# 4. Check RealVNC services
systemctl list-units | grep vnc

# 5. Check if desktop session exists
loginctl list-sessions
echo $XDG_SESSION_TYPE

# 6. Check X servers running
ps aux | grep X

# 7. Check VirtualGL installed
which vglrun
dpkg -l | grep virtualgl
```

---

## Known Working Alternative (Current Workaround)

**Run tests locally on desktop:**
```bash
# 1. Physically sit at desktop or be logged into GUI
# 2. Open terminal
# 3. Run test:
cd ~/agnostic-MAPush
source ./results/11-07-17_cuboid/task/train.sh True

# 4. Copy video to Mac via SSH:
scp gvlab@10.9.73.9:~/agnostic-MAPush/docs/video/test_seed*.mp4 ./
```

**Status:** ✅ This works perfectly
**Limitation:** Requires physical access or already being logged in

---

## Multi-Episode Recording Implementation (Working)

**Status:** ✅ **FULLY WORKING** when run locally

### Implementation Details
- **File:** `openrl_ws/test.py`
- **Config:** Hardcoded at top of file
  ```python
  SEED = 5
  NUM_EPISODES = 3
  ```
- **Method:** Monkey-patching at runtime
- **Output:** `docs/video/test_seed5_3eps.mp4`

### What Works
✅ Records N consecutive episodes in single video
✅ Seed control working
✅ Frame accumulation across episodes
✅ Proper episode counting and progress tracking
✅ Clean video output with correct duration

### Only Limitation
❌ Graphics rendering fails over plain SSH (blank frames)
✅ Works perfectly when run locally with display access

**Documentation:** See `MULTI_EPISODE_RECORDING.md` for full details

---

## References

- **Multi-episode recording docs:** `MULTI_EPISODE_RECORDING.md`
- **Setup commands attempted:** `vnc_setup_commands.txt`, `tigervnc_setup.txt`
- **Isaac Gym docs:** `/home/gvlab/isaac_gym/docs/`
- **RealVNC docs:** https://help.realvnc.com/hc/en-us/articles/360002253878
- **TigerVNC docs:** https://tigervnc.org/

---

## Session Context for Tomorrow

**Where We Left Off:**
- Tried multiple VNC approaches, all connection refused
- RealVNC Service Mode running but not functional (no desktop session)
- TigerVNC command hijacked by RealVNC
- No VNC server actually listening on any port
- Need to either: fix RealVNC Virtual Mode, or switch to TurboVNC with VirtualGL

**Priority Order for Tomorrow:**
1. Try RealVNC Virtual Mode properly (`vncserver-virtuald`)
2. If that fails, install TurboVNC + VirtualGL (best for GPU apps)
3. If still failing, troubleshoot why no ports are listening
4. Last resort: Set up automatic desktop login so Service Mode works

**Quick Win Test:**
If you can physically login to the desktop GUI tomorrow, RealVNC Service Mode should work immediately without any changes.

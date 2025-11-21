# How to Access TensorBoard Remotely

**For**: Viewing TensorBoard running on `gvlab-desktop` from your local computer
**Date**: 2025-11-19

---

## The Problem

TensorBoard runs on the remote server (`gvlab-desktop`) at `http://localhost:6006`, but you can't access it directly from your local browser because:
- "localhost" on the server ≠ "localhost" on your computer
- The port is not exposed to external connections by default

**Solution**: SSH Port Forwarding (SSH Tunnel)

---

## Quick Guide: 3 Steps

### Step 1: Start TensorBoard on the Server

On `gvlab-desktop` (via SSH), run:

```bash
cd /home/gvlab/agnostic-MAPush
./view_happo_comparison.sh
```

You should see:
```
TensorBoard 2.14.0 at http://gvlab-desktop:6006/ (Press CTRL+C to quit)
```

**Leave this running!** Don't close this terminal.

---

### Step 2: Set Up SSH Port Forwarding

On **your local computer** (laptop/desktop), open a **new terminal** and run:

```bash
ssh -L 6006:localhost:6006 gvlab@10.9.73.9
```

**What this does**:
- `-L 6006:localhost:6006` = Forward local port 6006 to server's port 6006
- `gvlab@10.9.73.9` = Connect to the server as user `gvlab`

**Leave this SSH connection open!** The port forwarding only works while connected.

---

### Step 3: Open TensorBoard in Your Browser

On **your local computer**, open your browser and go to:

```
http://localhost:6006
```

You should now see TensorBoard! 🎉

---

## Troubleshooting

### "Connection refused" or "Can't connect"

**Check 1**: Is TensorBoard still running on the server?
```bash
# On the server:
ps aux | grep tensorboard
```

If nothing shows up, restart it:
```bash
cd /home/gvlab/agnostic-MAPush
./view_happo_comparison.sh
```

**Check 2**: Is the SSH port forwarding still active?

Make sure your SSH terminal from Step 2 is still open and connected.

---

### "Port 6006 is already in use" (on server)

Someone else is using port 6006. Two options:

**Option A**: Kill the existing TensorBoard and restart
```bash
pkill -f tensorboard
./view_happo_comparison.sh
```

**Option B**: Use a different port
```bash
# Start TensorBoard on port 6007 instead
/home/gvlab/miniconda3/envs/mapush/bin/python -m tensorboard.main \
  --logdir results/mapush/go1push_mid/happo/cuboid \
  --port 6007

# Then use port 6007 in your SSH forwarding:
ssh -L 6007:localhost:6007 gvlab@10.9.73.9

# Open browser to:
http://localhost:6007
```

---

### "Port 6006 is already in use" (on local computer)

Something on your local computer is using port 6006.

**Solution**: Use a different local port:
```bash
# Forward local port 6007 to server's port 6006
ssh -L 6007:localhost:6006 gvlab@10.9.73.9

# Open browser to:
http://localhost:6007
```

---

## How It Works (Technical Details)

```
┌─────────────────┐                           ┌──────────────────┐
│ Your Computer   │                           │  gvlab-desktop   │
│ (Local)         │                           │  (Remote Server) │
│                 │                           │                  │
│  Browser        │   SSH Tunnel (Encrypted)  │  TensorBoard     │
│    ↓            │   ←─────────────────────→ │    ↓             │
│  localhost:6006 │                           │  localhost:6006  │
└─────────────────┘                           └──────────────────┘
```

**SSH Port Forwarding**:
1. You open `http://localhost:6006` in your browser
2. Your computer forwards this request through the SSH tunnel
3. The server receives it as if it came from `localhost:6006`
4. TensorBoard responds
5. Response travels back through the SSH tunnel
6. Your browser displays it

---

## Starting TensorBoard Manually

If `view_happo_comparison.sh` doesn't work, start TensorBoard manually:

### Compare Old vs New HAPPO Runs:
```bash
cd /home/gvlab/agnostic-MAPush

/home/gvlab/miniconda3/envs/mapush/bin/python -m tensorboard.main \
  --logdir_spec=old_happo_unstable:results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-15-15-34-13/logs,new_happo_safe:results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-19-13-26-03/logs \
  --port 6006 \
  --bind_all
```

### View Only New Training:
```bash
cd /home/gvlab/agnostic-MAPush

/home/gvlab/miniconda3/envs/mapush/bin/python -m tensorboard.main \
  --logdir results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-19-13-26-03/logs \
  --port 6006
```

### View All HAPPO Runs:
```bash
cd /home/gvlab/agnostic-MAPush

/home/gvlab/miniconda3/envs/mapush/bin/python -m tensorboard.main \
  --logdir results/mapush/go1push_mid/happo/cuboid \
  --port 6006
```

---

## Running TensorBoard in Background

If you want to close the terminal but keep TensorBoard running:

### Method 1: Using nohup
```bash
cd /home/gvlab/agnostic-MAPush

nohup /home/gvlab/miniconda3/envs/mapush/bin/python -m tensorboard.main \
  --logdir results/mapush/go1push_mid/happo/cuboid \
  --port 6006 \
  --bind_all > /tmp/tensorboard.log 2>&1 &

# Check if it started:
tail -f /tmp/tensorboard.log
# Press Ctrl+C to stop viewing the log (TensorBoard keeps running)
```

### Method 2: Using screen (recommended)
```bash
# Install screen if not available:
sudo apt install screen

# Start a screen session:
screen -S tensorboard

# Inside screen, start TensorBoard:
cd /home/gvlab/agnostic-MAPush
./view_happo_comparison.sh

# Detach from screen: Press Ctrl+A then D
# TensorBoard keeps running in background

# Reattach later:
screen -r tensorboard

# Kill screen session:
screen -X -S tensorboard quit
```

### Stop Background TensorBoard:
```bash
# Kill all TensorBoard processes:
pkill -f tensorboard

# Or kill specific port:
lsof -ti:6006 | xargs kill -9
```

---

## Alternative: VS Code Port Forwarding

If you're using **VS Code Remote SSH**:

1. Start TensorBoard on server (Step 1 above)
2. In VS Code:
   - Open Command Palette: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type: "Forward a Port"
   - Enter: `6006`
3. VS Code will automatically forward the port
4. Click the link VS Code shows, or go to `http://localhost:6006`

**Advantage**: Simpler, no need for manual SSH command!

---

## Cheat Sheet

```bash
# === ON SERVER (gvlab-desktop) ===

# Start TensorBoard:
cd /home/gvlab/agnostic-MAPush
./view_happo_comparison.sh

# Or manually:
/home/gvlab/miniconda3/envs/mapush/bin/python -m tensorboard.main \
  --logdir results/mapush/go1push_mid/happo/cuboid --port 6006

# Check if TensorBoard is running:
ps aux | grep tensorboard

# Kill TensorBoard:
pkill -f tensorboard


# === ON LOCAL COMPUTER ===

# Set up SSH port forwarding:
ssh -L 6006:localhost:6006 gvlab@10.9.73.9

# Then open browser to:
# http://localhost:6006
```

---

## FAQ

**Q: Do I need to keep the SSH terminal open?**
A: Yes, the port forwarding only works while the SSH connection is active.

**Q: Can multiple people access TensorBoard at the same time?**
A: Yes! Each person sets up their own SSH port forwarding. TensorBoard supports multiple viewers.

**Q: What if the server IP changes?**
A: Find the new IP with:
```bash
# On the server:
hostname -I | awk '{print $1}'
```

**Q: Can I access TensorBoard from my phone?**
A: Not easily with SSH port forwarding. You'd need to:
1. Set up TensorBoard to bind to all interfaces: `--bind_all`
2. Allow port 6006 through firewall
3. Access via `http://10.9.73.9:6006`
4. (Not recommended for security reasons)

**Q: Is this secure?**
A: Yes! SSH port forwarding is encrypted. All data travels through the secure SSH tunnel.

---

**Created**: 2025-11-19
**Server IP**: 10.9.73.9
**Default Port**: 6006

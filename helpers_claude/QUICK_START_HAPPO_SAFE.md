# Quick Start: HAPPO Safe Training

**Status**: ✅ Ready to go!
**Date**: 2025-11-19

---

## 🚀 Start Training (One Command)

```bash
cd /home/gvlab/agnostic-MAPush
source task/cuboid/train.sh False
```

That's it! Training will start with the safe HAPPO configuration.

---

## 📊 Monitor Progress (TensorBoard)

In a **separate terminal**:

```bash
cd /home/gvlab/agnostic-MAPush
./view_happo_comparison.sh
```

Then open: **http://localhost:6006**

---

## ⚙️ What's Running

- **Algorithm**: HAPPO with safe config
- **Task**: 2 × Go1 robots pushing cuboid
- **Environments**: 500 parallel
- **Total steps**: 100M
- **Checkpoint frequency**: Every 2.5M steps

---

## 🎯 What to Watch (First Hour)

In TensorBoard, check:

1. **`train_episode_rewards/aver_rewards`**
   - Should start around -15.0
   - Look for steady improvement

2. **`agent0/actor_grad_norm`** and **`agent1/actor_grad_norm`**
   - Should be 0.2-0.5 (healthy range)
   - Max is clipped at 0.5 (safety feature)

3. **`critic/value_loss`**
   - Should decrease over time

4. **`agent0/dist_entropy`**
   - Should start ~2.0, slowly decrease

---

## ⚠️ Red Flags

Stop and investigate if you see:

- Rewards diverging or becoming positive
- Value loss increasing instead of decreasing
- Entropy dropping to 0 quickly (< 5M steps)
- Training crashes with gradient errors

---

## 📁 Where Results Are Saved

```
results/mapush/go1push_mid/happo/cuboid/seed-00001-<timestamp>/
├── models/
│   ├── rl_model_2500000_steps/
│   ├── rl_model_5000000_steps/
│   └── ... (every 2.5M steps)
├── logs/ (TensorBoard logs)
├── config.json
└── progress.txt
```

---

## 🔍 Test a Checkpoint (While Training)

```bash
# Wait for first checkpoint (2.5M steps, ~30-45 min), then:
source task/cuboid/train.sh True
```

This will test the latest checkpoint in viewer mode.

---

## ⏱️ Expected Timeline

- **First checkpoint**: 2.5M steps (~30-45 min)
- **Early evaluation**: 10M steps (~2-3 hours)
- **Mid-point**: 50M steps (~9-12 hours)
- **Full training**: 100M steps (~18-24 hours)

---

## 🛑 Stop Training

Just press `Ctrl+C` in the training terminal.

Checkpoints are saved automatically every 2.5M steps, so you can resume later if needed.

---

## ✅ Success Criteria

Training is successful if:

1. **No crashes** - Runs stably for 100M steps
2. **No mid-training degradation** - Unlike old run that degraded at 50-75M
3. **Final reward** > -10.0 (good) or > -5.0 (excellent)
4. **Gradient norms** stay healthy (0.2-0.5 range)

---

## 📖 Full Documentation

See `TRAINING_SETUP_READY.md` for:
- Detailed monitoring instructions
- Troubleshooting guide
- Configuration details
- How to restore original HAPPO config

---

## 🔧 What Was Fixed

The error you saw (`invalid choice: 'happo_safe'`) was because HARL's train.py only accepts algorithm names that match config files.

**Solution**: Replaced `happo.yaml` with the safe version. Original backed up as `happo_original.yaml`.

---

**Ready to train!** Just run: `source task/cuboid/train.sh False`

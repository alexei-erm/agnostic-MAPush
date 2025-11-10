# Checkpoint Testing Script

Automated script to test multiple model checkpoints in calculator mode.

## Files

- **test_checkpoints.sh** - Main testing script
- **test_checkpoints_example.sh** - Interactive helper with examples
- **TEST_CHECKPOINTS_README.md** - This file

## Quick Start

```bash
# Make sure you're in the project root directory
cd /home/gvlab/newMAPush

# Test checkpoints 10M through 100M (every 10M)
./test_checkpoints.sh ./results/10-15-23_cylinder 10 20 30 40 50 60 70 80 90 100
```

## Usage

```bash
./test_checkpoints.sh <results_directory> <checkpoint_numbers...>
```

### Arguments

- **results_directory**: Path to the results folder containing checkpoints
  - Example: `./results/10-15-23_cylinder`
  - Must contain a `checkpoints/` subdirectory

- **checkpoint_numbers**: Space-separated list of checkpoint numbers (in millions)
  - `10` = 10,000,000 steps
  - `20` = 20,000,000 steps
  - `100` = 100,000,000 steps
  - etc.

## Examples

### Test all checkpoints (10M to 100M, every 10M)
```bash
./test_checkpoints.sh ./results/10-15-23_cylinder 10 20 30 40 50 60 70 80 90 100
```

### Test specific checkpoints only
```bash
./test_checkpoints.sh ./results/10-15-23_cylinder 20 50 100
```

### Test with finer granularity (every 5M)
```bash
./test_checkpoints.sh ./results/10-15-23_cylinder 5 10 15 20 25 30 35 40 45 50
```

### Test cuboid with 110M checkpoint
```bash
./test_checkpoints.sh ./results/11-06-17_cuboid 10 20 30 40 50 60 70 80 90 100 110
```

### Quick test - first and last only
```bash
./test_checkpoints.sh ./results/10-15-23_cylinder 10 100
```

## Interactive Helper

For guided usage with examples:

```bash
./test_checkpoints_example.sh
```

This will show examples and let you choose from common testing scenarios.

## Output

Results are appended to `<results_directory>/success_rate.txt`

The output includes:
- Success rate
- Finished time
- Collision degree
- Collaboration degree

Each checkpoint's results are separated by a line of dashes.

## Features

- ✅ Tests multiple checkpoints in sequence
- ✅ Continues testing even if one checkpoint fails
- ✅ Validates directory and checkpoint file existence
- ✅ Auto-detects algorithm from train.sh
- ✅ Provides detailed progress output
- ✅ Summary statistics at the end
- ✅ Confirmation prompt before starting

## Requirements

- Must be run from project root directory (`/home/gvlab/newMAPush`)
- Python environment must be activated
- IsaacGym and dependencies must be available
- Results directory must have proper structure:
  ```
  results/<experiment>/
  ├── checkpoints/
  │   ├── rl_model_10000000_steps/
  │   │   └── module.pt
  │   ├── rl_model_20000000_steps/
  │   │   └── module.pt
  │   └── ...
  └── task/
      ├── config.py
      └── train.sh
  ```

## Troubleshooting

### "Directory does not exist"
- Check the path is correct
- Use absolute or relative path from project root

### "Checkpoint file does not exist"
- Verify checkpoint numbers match actual saved checkpoints
- Check the `checkpoints/` subdirectory structure

### Test fails with dimension mismatch
- Checkpoint was trained with different environment configuration
- Check task configuration matches training configuration
- Script will skip and continue to next checkpoint

### Script doesn't have execute permissions
```bash
chmod +x test_checkpoints.sh
chmod +x test_checkpoints_example.sh
```

## Notes

- Numbers are in millions: `10` = `10000000` steps, `50` = `50000000` steps
- Output is appended, not overwritten (allows multiple test runs)
- Each test uses 300 environments (`--num_envs 300`)
- Tests run in headless mode (no GUI)
- Calculator mode computes metrics across all environments

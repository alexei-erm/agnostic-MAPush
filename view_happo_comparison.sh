#!/bin/bash
# TensorBoard script to compare multiple HAPPO runs
# Automatically discovers and displays all runs in a directory

# Default directory containing the runs
DEFAULT_DIR="results/mapush/go1push_mid/happo/cuboid"

# Allow user to specify a different directory as an argument
RUNS_DIR="${1:-$DEFAULT_DIR}"

echo "Starting TensorBoard to compare HAPPO runs..."
echo ""
echo "Searching for runs in: $RUNS_DIR"
echo ""

# Check if directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Directory '$RUNS_DIR' not found!"
    echo "Usage: $0 [directory_path]"
    echo "Example: $0 results/mapush/go1push_mid/happo/cuboid"
    exit 1
fi

# Find all seed directories with logs
RUN_DIRS=$(find "$RUNS_DIR" -maxdepth 1 -type d -name "seed-*" | sort)

# Count the number of runs found
RUN_COUNT=$(echo "$RUN_DIRS" | grep -c "seed-" 2>/dev/null || echo "0")

if [ "$RUN_COUNT" -eq 0 ]; then
    echo "Error: No training runs found in $RUNS_DIR"
    echo "Expected directories matching pattern: seed-*"
    exit 1
fi

echo "Found $RUN_COUNT run(s):"
echo ""

# Build the logdir_spec for TensorBoard
LOGDIR_SPEC=""
COUNTER=1

while IFS= read -r run_dir; do
    if [ -d "$run_dir/logs" ]; then
        # Extract timestamp from directory name for labeling
        RUN_NAME=$(basename "$run_dir" | sed 's/seed-[0-9]*-//')

        echo "  $COUNTER. $RUN_NAME"
        echo "     Path: $run_dir"

        # Add to logdir_spec
        if [ -z "$LOGDIR_SPEC" ]; then
            LOGDIR_SPEC="run_${COUNTER}_${RUN_NAME}:$run_dir/logs"
        else
            LOGDIR_SPEC="${LOGDIR_SPEC},run_${COUNTER}_${RUN_NAME}:$run_dir/logs"
        fi

        COUNTER=$((COUNTER + 1))
    fi
done <<< "$RUN_DIRS"

echo ""
echo "TensorBoard will be available at: http://localhost:6006"
echo "Press Ctrl+C to stop TensorBoard"
echo ""

# Launch TensorBoard with all discovered runs
tensorboard --logdir_spec="$LOGDIR_SPEC" --port 6006

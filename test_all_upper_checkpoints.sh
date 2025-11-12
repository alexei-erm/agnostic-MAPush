#!/bin/bash

# Script to test all upper-level checkpoints (excluding 100M)
# Run in tmux session for overnight testing

CHECKPOINT_DIR="./results/models/baseline_mappo/upper/checkpoints"
LOG_FILE="./upper_checkpoints_test_$(date +%Y%m%d_%H%M%S).log"
RESULTS_FILE="./upper_success_rates.txt"

# Clear/create results file
echo "=== Upper-Level Controller Success Rates ===" > "$RESULTS_FILE"
echo "Test started at: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Checkpoint steps to test (excluding 100M)
CHECKPOINT_STEPS=(10000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000)

echo "Starting upper-level checkpoint testing at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to: $RESULTS_FILE" | tee -a "$LOG_FILE"
echo "Full log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counter for progress
TOTAL=${#CHECKPOINT_STEPS[@]}
CURRENT=0

for steps in "${CHECKPOINT_STEPS[@]}"; do
    CURRENT=$((CURRENT + 1))
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/rl_model_${steps}_steps/module.pt"

    echo "[$CURRENT/$TOTAL] Testing checkpoint: ${steps} steps" | tee -a "$LOG_FILE"
    echo "Checkpoint path: $CHECKPOINT_PATH" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"

    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH" | tee -a "$LOG_FILE"
        echo "Skipping..." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    # Run the test
    echo "Running test..." | tee -a "$LOG_FILE"
    python ./openrl_ws/test.py \
        --algo ppo \
        --test_mode calculator \
        --headless \
        --task go1push_upper \
        --num_envs 300 \
        --checkpoint "$CHECKPOINT_PATH" 2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Test completed successfully" | tee -a "$LOG_FILE"

        # Try to extract success rate from log
        SUCCESS_RATE=$(tail -100 "$LOG_FILE" | grep -i "success rate" | tail -1)
        if [ -n "$SUCCESS_RATE" ]; then
            echo "${steps} steps: $SUCCESS_RATE" >> "$RESULTS_FILE"
        else
            echo "${steps} steps: Test completed (check log for details)" >> "$RESULTS_FILE"
        fi
    else
        echo "✗ Test failed with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
        echo "${steps} steps: FAILED (exit code $EXIT_CODE)" >> "$RESULTS_FILE"
    fi

    echo "Completed at: $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "All tests completed at: $(date)" | tee -a "$LOG_FILE"
echo "" >> "$RESULTS_FILE"
echo "Test completed at: $(date)" >> "$RESULTS_FILE"

# Display final results
echo "" | tee -a "$LOG_FILE"
echo "=== FINAL RESULTS ===" | tee -a "$LOG_FILE"
cat "$RESULTS_FILE" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results summary saved to: $RESULTS_FILE" | tee -a "$LOG_FILE"

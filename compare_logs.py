#!/usr/bin/env python3
"""
Compare training statistics between baseline MAPPO and HAPPO runs
"""

import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np

def load_tensorboard_data(logdir, tags=None):
    """Load data from TensorBoard event files"""
    event_files = glob.glob(os.path.join(logdir, "**/events.out.tfevents.*"), recursive=True)

    if not event_files:
        return None

    data = {}

    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # Get the metric name from the parent directory
        metric_name = os.path.basename(os.path.dirname(event_file))

        # Get scalar data
        for tag in ea.Tags().get('scalars', []):
            scalar_events = ea.Scalars(tag)
            if scalar_events:
                steps = [e.step for e in scalar_events]
                values = [e.value for e in scalar_events]
                full_tag = f"{metric_name}/{tag}" if metric_name != "." else tag
                data[full_tag] = {'steps': steps, 'values': values}

    return data

def summarize_metrics(data, name):
    """Summarize metrics from tensorboard data"""
    print(f"\n{'='*60}")
    print(f"{name} - Training Metrics Summary")
    print(f"{'='*60}\n")

    if not data:
        print("No data found!")
        return

    for metric, values in sorted(data.items()):
        if values['values']:
            vals = np.array(values['values'])
            print(f"{metric}:")
            print(f"  Initial: {vals[0]:.6f}")
            print(f"  Final:   {vals[-1]:.6f}")
            print(f"  Mean:    {vals.mean():.6f}")
            print(f"  Std:     {vals.std():.6f}")
            print(f"  Min:     {vals.min():.6f}")
            print(f"  Max:     {vals.max():.6f}")
            print(f"  Steps:   {len(vals)}")
            print()

def main():
    # Paths
    mappo_logdir = "./results/models/baseline_mappo/mid/logs"
    happo_logdir = "./results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-19-13-26-03/logs"

    print("Loading MAPPO data...")
    mappo_data = load_tensorboard_data(mappo_logdir)

    print("Loading HAPPO data...")
    happo_data = load_tensorboard_data(happo_logdir)

    # Summarize
    summarize_metrics(mappo_data, "Baseline MAPPO")
    summarize_metrics(happo_data, "HAPPO")

    # Compare common metrics
    print(f"\n{'='*60}")
    print("Comparison of Common Metrics")
    print(f"{'='*60}\n")

    if mappo_data and happo_data:
        mappo_keys = set(mappo_data.keys())
        happo_keys = set(happo_data.keys())

        print("MAPPO metrics:", len(mappo_keys))
        print("HAPPO metrics:", len(happo_keys))

        print("\nMAPPO-specific metrics:")
        for key in sorted(mappo_keys - happo_keys):
            print(f"  - {key}")

        print("\nHAPPO-specific metrics:")
        for key in sorted(happo_keys - mappo_keys):
            print(f"  - {key}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple comparison of MAPPO vs HAPPO training logs
"""

import re
import os

def parse_mappo_log(logfile):
    """Parse MAPPO log.txt file"""
    metrics = {
        'average_step_reward': [],
        'reach_target_reward': [],
        'distance_to_target_reward': [],
        'collision_punishment': [],
        'push_reward': [],
        'value_loss': [],
        'policy_loss': [],
        'dist_entropy': [],
        'actor_grad_norm': [],
        'critic_grad_norm': [],
        'ratio': []
    }

    with open(logfile, 'r') as f:
        content = f.read()

    # Extract metrics
    for key in metrics.keys():
        pattern = rf'{key}:\s*([-\d.e]+)'
        matches = re.findall(pattern, content)
        metrics[key] = [float(m) for m in matches]

    return metrics

def print_stats(metrics, name):
    """Print statistics for metrics"""
    print(f"\n{'='*70}")
    print(f"{name} Training Statistics")
    print(f"{'='*70}")

    if not metrics:
        print("No metrics found!")
        return

    for key, values in metrics.items():
        if values:
            print(f"\n{key}:")
            print(f"  Episodes: {len(values)}")
            print(f"  Initial:  {values[0]:.6f}")
            print(f"  Final:    {values[-1]:.6f}")
            print(f"  Mean:     {sum(values)/len(values):.6f}")

            # Calculate std manually
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = variance ** 0.5
            print(f"  Std:      {std:.6f}")
            print(f"  Min:      {min(values):.6f}")
            print(f"  Max:      {max(values):.6f}")

            # Show trend
            if len(values) >= 2:
                trend = values[-1] - values[0]
                trend_pct = (trend / abs(values[0])) * 100 if values[0] != 0 else 0
                print(f"  Change:   {trend:+.6f} ({trend_pct:+.2f}%)")

def compare_final_metrics(mappo_metrics, happo_metrics):
    """Compare final values between MAPPO and HAPPO"""
    print(f"\n{'='*70}")
    print("Final Metrics Comparison (MAPPO vs HAPPO)")
    print(f"{'='*70}")

    common_keys = set(mappo_metrics.keys()) & set(happo_metrics.keys())

    print(f"\n{'Metric':<30} {'MAPPO':>15} {'HAPPO':>15} {'Diff':>10}")
    print("-" * 70)

    for key in sorted(common_keys):
        if mappo_metrics[key] and happo_metrics[key]:
            mappo_final = mappo_metrics[key][-1]
            happo_final = happo_metrics[key][-1]
            diff = happo_final - mappo_final
            print(f"{key:<30} {mappo_final:>15.6f} {happo_final:>15.6f} {diff:>+10.6f}")

def main():
    # Paths
    mappo_log = "./results/models/baseline_mappo/mid/log.txt"

    # Check if MAPPO log exists
    if not os.path.exists(mappo_log):
        print(f"MAPPO log not found: {mappo_log}")
        return

    print("Parsing MAPPO logs...")
    mappo_metrics = parse_mappo_log(mappo_log)

    print_stats(mappo_metrics, "MAPPO (baseline)")

    # For HAPPO, we need to check what logs are available
    happo_runs = [
        "./results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-19-13-26-03",
        "./results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-15-15-34-13"
    ]

    print(f"\n\n{'='*70}")
    print("HAPPO Runs Information")
    print(f"{'='*70}")

    for run_dir in happo_runs:
        if os.path.exists(run_dir):
            print(f"\nRun: {os.path.basename(run_dir)}")
            # List what's in the directory
            print(f"  Directory: {run_dir}")
            if os.path.exists(os.path.join(run_dir, "logs")):
                log_subdirs = os.listdir(os.path.join(run_dir, "logs"))
                print(f"  Log subdirs: {', '.join(log_subdirs)}")
            if os.path.exists(os.path.join(run_dir, "models")):
                models = os.listdir(os.path.join(run_dir, "models"))
                print(f"  Models: {len(models)} checkpoints")

            # Check config
            config_file = os.path.join(run_dir, "config.json")
            if os.path.exists(config_file):
                import json
                with open(config_file) as f:
                    config = json.load(f)
                print(f"  Algorithm: {config.get('main_args', {}).get('algo', 'unknown')}")
                print(f"  Num envs: {config.get('env_args', {}).get('num_envs', 'unknown')}")
                print(f"  Episode length: {config.get('env_args', {}).get('episode_length', 'unknown')}")
                print(f"  Total steps: {config.get('algo_args', {}).get('train', {}).get('num_env_steps', 'unknown')}")

if __name__ == "__main__":
    main()

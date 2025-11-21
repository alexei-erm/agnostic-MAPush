#!/usr/bin/env python3
"""
Compare MAPPO vs HAPPO training performance
"""

import json
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

def parse_happo_summary(summary_file):
    """Parse HAPPO summary.json file"""
    with open(summary_file, 'r') as f:
        data = json.load(f)

    # Organize metrics by agent
    agent0_metrics = {}
    agent1_metrics = {}
    critic_metrics = {}
    reward_metrics = {}

    for path, values in data.items():
        # Extract just the values (third element of each entry)
        metric_values = [v[2] for v in values]

        if '/agent0/' in path:
            metric_name = path.split('/')[-1]
            agent0_metrics[metric_name] = metric_values
        elif '/agent1/' in path:
            metric_name = path.split('/')[-1]
            agent1_metrics[metric_name] = metric_values
        elif '/critic/' in path:
            metric_name = path.split('/')[-1]
            critic_metrics[metric_name] = metric_values
        elif '/train_episode_rewards/' in path:
            metric_name = path.split('/')[-1]
            reward_metrics[metric_name] = metric_values

    return {
        'agent0': agent0_metrics,
        'agent1': agent1_metrics,
        'critic': critic_metrics,
        'rewards': reward_metrics
    }

def print_comparison_table(mappo_metrics, happo_metrics):
    """Print comparison table of final metrics"""
    print(f"\n{'='*80}")
    print("TRAINING COMPARISON: MAPPO vs HAPPO")
    print(f"{'='*80}\n")

    print("MAPPO Configuration:")
    print("  - Centralized training, centralized execution")
    print("  - Single shared policy\n")

    print("HAPPO Configuration:")
    print("  - Heterogeneous-Agent PPO")
    print("  - Separate policies per agent (agent0, agent1)")
    print("  - Shared critic\n")

    # MAPPO final metrics
    print(f"\n{'-'*80}")
    print("MAPPO Final Performance")
    print(f"{'-'*80}")
    key_metrics = ['average_step_reward', 'reach_target_reward', 'push_reward',
                   'collision_punishment', 'value_loss', 'policy_loss', 'dist_entropy']

    for key in key_metrics:
        if key in mappo_metrics and mappo_metrics[key]:
            initial = mappo_metrics[key][0]
            final = mappo_metrics[key][-1]
            change = final - initial
            print(f"{key:30s}: {final:10.6f}  (change: {change:+.6f})")

    # HAPPO final metrics
    print(f"\n{'-'*80}")
    print("HAPPO Final Performance")
    print(f"{'-'*80}")

    if 'agent0' in happo_metrics:
        print("\nAgent 0 Metrics:")
        for key, values in happo_metrics['agent0'].items():
            if values:
                initial = values[0]
                final = values[-1]
                change = final - initial
                print(f"  {key:30s}: {final:10.6f}  (change: {change:+.6f})")

    if 'agent1' in happo_metrics:
        print("\nAgent 1 Metrics:")
        for key, values in happo_metrics['agent1'].items():
            if values:
                initial = values[0]
                final = values[-1]
                change = final - initial
                print(f"  {key:30s}: {final:10.6f}  (change: {change:+.6f})")

    if 'critic' in happo_metrics:
        print("\nCritic Metrics:")
        for key, values in happo_metrics['critic'].items():
            if values:
                initial = values[0]
                final = values[-1]
                change = final - initial
                print(f"  {key:30s}: {final:10.6f}  (change: {change:+.6f})")

    if 'rewards' in happo_metrics:
        print("\nReward Metrics:")
        for key, values in happo_metrics['rewards'].items():
            if values:
                initial = values[0]
                final = values[-1]
                change = final - initial
                mean = sum(values) / len(values)
                print(f"  {key:30s}: {final:10.6f}  (mean: {mean:.6f}, change: {change:+.6f})")

    # Comparison
    print(f"\n{'='*80}")
    print("KEY COMPARISONS")
    print(f"{'='*80}")

    print(f"\n{'Metric':<35} {'MAPPO':>15} {'HAPPO':>15} {'Difference':>15}")
    print(f"{'-'*80}")

    # Compare average reward
    if 'average_step_reward' in mappo_metrics and mappo_metrics['average_step_reward']:
        mappo_reward = mappo_metrics['average_step_reward'][-1]
        if 'aver_rewards' in happo_metrics.get('rewards', {}):
            happo_reward = happo_metrics['rewards']['aver_rewards'][-1]
            diff = happo_reward - mappo_reward
            print(f"{'Average Episode Reward':<35} {mappo_reward:>15.6f} {happo_reward:>15.6f} {diff:>+15.6f}")

    # Compare entropy
    if 'dist_entropy' in mappo_metrics and mappo_metrics['dist_entropy']:
        mappo_entropy = mappo_metrics['dist_entropy'][-1]
        if 'dist_entropy' in happo_metrics.get('agent0', {}):
            # Average entropy from both agents
            agent0_entropy = happo_metrics['agent0']['dist_entropy'][-1]
            agent1_entropy = happo_metrics['agent1']['dist_entropy'][-1] if 'dist_entropy' in happo_metrics.get('agent1', {}) else agent0_entropy
            happo_entropy = (agent0_entropy + agent1_entropy) / 2
            diff = happo_entropy - mappo_entropy
            print(f"{'Policy Entropy (exploration)':<35} {mappo_entropy:>15.6f} {happo_entropy:>15.6f} {diff:>+15.6f}")

    # Compare value loss
    if 'value_loss' in mappo_metrics and mappo_metrics['value_loss']:
        mappo_vloss = mappo_metrics['value_loss'][-1]
        if 'value_loss' in happo_metrics.get('critic', {}):
            happo_vloss = happo_metrics['critic']['value_loss'][-1]
            diff = happo_vloss - mappo_vloss
            print(f"{'Value Loss':<35} {mappo_vloss:>15.6f} {happo_vloss:>15.6f} {diff:>+15.6f}")

    print("\n")

def main():
    # Paths
    mappo_log = "./results/models/baseline_mappo/mid/log.txt"
    happo_summary = "./results/mapush/go1push_mid/happo/cuboid/seed-00001-2025-11-19-13-26-03/logs/summary.json"

    # Check files
    if not os.path.exists(mappo_log):
        print(f"ERROR: MAPPO log not found: {mappo_log}")
        return

    if not os.path.exists(happo_summary):
        print(f"ERROR: HAPPO summary not found: {happo_summary}")
        return

    print("Parsing logs...")
    mappo_metrics = parse_mappo_log(mappo_log)
    happo_metrics = parse_happo_summary(happo_summary)

    print_comparison_table(mappo_metrics, happo_metrics)

    print("\nNOTES:")
    print("- MAPPO uses a single shared policy for all agents")
    print("- HAPPO uses separate policies for heterogeneous agents (more flexible)")
    print("- Higher entropy = more exploration (can be good or bad depending on stage)")
    print("- Lower value loss = better value function estimation")
    print("- Reward improvement shows learning effectiveness\n")

if __name__ == "__main__":
    main()

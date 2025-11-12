"""Test script for MAPush HARL environment wrapper."""
import sys
import os

# CRITICAL: Import isaacgym BEFORE any other modules that might import torch
import isaacgym

import numpy as np

# Add HARL to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'HARL'))

def test_mapush_env():
    """Test MAPush environment wrapper with basic reset and step operations."""
    print("=" * 60)
    print("Testing MAPush HARL Environment Wrapper")
    print("=" * 60)

    try:
        # Import after path is set
        from harl.envs.mapush.mapush_env import MAPushEnv
        print("✓ Successfully imported MAPushEnv")

        # Create environment with minimal number of envs for testing
        print("\nCreating environment with 10 parallel environments...")
        env_args = {
            "task": "go1push_mid",
            "num_envs": 10,
            "seed": 1,
            "headless": True,
            "device": "cuda:0",
        }

        env = MAPushEnv(env_args)
        print(f"✓ Environment created successfully")
        print(f"  - Number of parallel envs: {env.n_envs}")
        print(f"  - Number of agents: {env.n_agents}")
        print(f"  - Observation space: {env.observation_space[0].shape}")
        print(f"  - State space: {env.share_observation_space[0].shape}")
        print(f"  - Action space: {env.action_space[0].shape}")

        # Test reset
        print("\nTesting reset()...")
        obs, state, avail_actions = env.reset()
        print(f"✓ Reset successful")
        print(f"  - Observations shape: {obs.shape}")
        print(f"  - State shape: {state.shape}")
        print(f"  - Available actions: {avail_actions}")

        # Validate shapes
        assert obs.shape == (env.n_envs, env.n_agents, env.observation_space[0].shape[0]), \
            f"Observation shape mismatch: {obs.shape}"
        assert state.shape == (env.n_envs, env.n_agents, env.share_observation_space[0].shape[0]), \
            f"State shape mismatch: {state.shape}"
        print("✓ Shapes validated")

        # Test step with random actions
        print("\nTesting step() with random actions...")
        actions = np.random.randn(env.n_envs, env.n_agents, env.action_space[0].shape[0])
        obs, state, rewards, dones, infos, avail_actions = env.step(actions)
        print(f"✓ Step successful")
        print(f"  - Observations shape: {obs.shape}")
        print(f"  - State shape: {state.shape}")
        print(f"  - Rewards shape: {rewards.shape}")
        print(f"  - Dones shape: {dones.shape}")
        print(f"  - Number of info dicts: {len(infos)} x {len(infos[0])}")

        # Validate shapes
        assert obs.shape == (env.n_envs, env.n_agents, env.observation_space[0].shape[0]), \
            f"Observation shape mismatch: {obs.shape}"
        assert state.shape == (env.n_envs, env.n_agents, env.share_observation_space[0].shape[0]), \
            f"State shape mismatch: {state.shape}"
        assert rewards.shape == (env.n_envs, env.n_agents, 1), \
            f"Rewards shape mismatch: {rewards.shape}"
        assert dones.shape == (env.n_envs, env.n_agents), \
            f"Dones shape mismatch: {dones.shape}"
        print("✓ Shapes validated")

        # Test multiple steps
        print("\nTesting 10 consecutive steps...")
        for i in range(10):
            actions = np.random.randn(env.n_envs, env.n_agents, env.action_space[0].shape[0])
            obs, state, rewards, dones, infos, avail_actions = env.step(actions)
        print(f"✓ All steps successful")

        # Test close
        print("\nClosing environment...")
        env.close()
        print("✓ Environment closed")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe MAPush HARL environment wrapper is working correctly.")
        print("You can now proceed with full training using:")
        print("  source task/cuboid/train.sh False")

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mapush_env()
    sys.exit(0 if success else 1)

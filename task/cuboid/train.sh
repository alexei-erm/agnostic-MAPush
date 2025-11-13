exp_name="cuboid"
current_dir=$(pwd)
algo="happo"  # Options: "happo", "hatrpo", "haa2c", "mappo"
script_path=$(realpath "${BASH_SOURCE[0]}")
# script_path=$(realpath $0)
script_dir=$(dirname "$script_path")
test_mode=$1

# update config
python ./openrl_ws/update_config.py --filepath $script_dir/config.py

if [ $test_mode = False ]; then
    # train
    num_envs=500
    num_steps=100000000
    checkpoint=/None  # "/results/07-28-13_task1/checkpoints/rl_model_100000000_steps/module.pt"

    # Check if using HARL algorithms (HAPPO, HATRPO, HAA2C) or OpenRL (MAPPO)
    if [ "$algo" = "happo" ] || [ "$algo" = "hatrpo" ] || [ "$algo" = "haa2c" ]; then
        # Use HARL for heterogeneous-agent algorithms
        echo "Training with HARL algorithm: $algo"
        # Add HARL to PYTHONPATH and run from project root to maintain correct relative paths
        PYTHONPATH="${current_dir}/HARL:${PYTHONPATH}" python HARL/examples/train.py \
            --algo "$algo" \
            --env mapush \
            --exp_name "$exp_name" \
            --seed 1 \
            --n_rollout_threads "$num_envs" \
            --num_env_steps "$num_steps" \
            --episode_length 200 \
            --log_interval 5 \
            --hidden_sizes "[256,256]" \
            --task go1push_mid \
            --headless True
    else
        # Use OpenRL for MAPPO
        echo "Training with OpenRL algorithm: $algo"
        python ./openrl_ws/train.py  --num_envs $num_envs --train_timesteps $num_steps\
        --algo $algo \
        --config ./openrl_ws/cfgs/ppo.yaml \
        --seed 1 \
        --exp_name  $exp_name \
        --task go1push_mid \
        --use_tensorboard \
        --checkpoint $current_dir$checkpoint \
        --headless \
        --hidden_size 256 \
        --layer_N 2
    fi

    # Calculate success rate (only for OpenRL/MAPPO - HARL has different checkpoint structure)
    if [ "$algo" = "mappo" ]; then
        steps=()
        for ((i=1; i<=num_steps/10000000; i++)); do
            steps+=("${i}0000000")
        done
        target_dir=$current_dir/results
        # Get last folder, excluding 'models' directory
        last_folder=$(ls -d $target_dir/*/ | grep -v "models/$" | sort | tail -n 1)
        echo "last_folder: $last_folder"
        if [ -d "$last_folder" ]; then
            for step in "${steps[@]}"; do
                filename="rl_model_${step}_steps/module.pt"
                test_checkpoint="$last_folder/checkpoints/$filename"
                if [ -f "$test_checkpoint" ]; then
                    python ./openrl_ws/test.py --num_envs 300 \
                            --algo "$algo" \
                            --task go1push_mid \
                            --checkpoint "$test_checkpoint" \
                            --test_mode calculator \
                            --headless  >> $last_folder/success_rate.txt 2>&1
                fi
            done
        fi
    else
        echo "Checkpoint testing for HARL algorithms will be done separately"
        echo "Checkpoints are saved in: HARL/results/mapush/go1push_mid/$algo/"
    fi

else
# test
root_dir=$(dirname "$script_dir")
filename="rl_model_110000000_steps/module.pt"
test_checkpoint="$root_dir/checkpoints/$filename"
python ./openrl_ws/test.py --num_envs 1 \
        --algo "$algo" \
        --task go1push_mid \
        --checkpoint "$test_checkpoint" \
        --test_mode viewer \
#       --record_video
fi

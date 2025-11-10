import isaacgym
import torch

# for upper
# checkpoint = torch.load('/home/gvlab/MAPush/log/MQE/go1push_upper/cuboid/run1/checkpoints/rl_model_100000000_steps/module.pt', map_location='cpu')
checkpoint = torch.load('/home/gvlab/newMAPush/results/11-06-17_cuboid/checkpoints/rl_model_10000000_steps/module.pt', map_location='cpu')

print(f"hidden_size: {checkpoint.cfg.hidden_size}")
print(f"layer_N: {checkpoint.cfg.layer_N}")
print(f"activation: {['Tanh', 'ReLU', 'LeakyReLU', 'ELU'][checkpoint.cfg.activation_id]}")
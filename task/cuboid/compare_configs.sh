#!/bin/bash
# Quick script to compare HAPPO configs side-by-side

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║            HAPPO Configuration Comparison                              ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Parameter Comparison:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-25s | %-15s | %-15s\n" "Parameter" "happo.yaml" "happo_safe.yaml"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract and compare key parameters
lr_old=$(grep "^  lr:" ../../HARL/harl/configs/algos_cfgs/happo.yaml | awk '{print $2}')
lr_new=$(grep "^  lr:" ../../HARL/harl/configs/algos_cfgs/happo_safe.yaml | awk '{print $2}')
printf "%-25s | %-15s | %-15s\n" "Learning Rate" "$lr_old" "$lr_new ⬇️"

critic_lr_old=$(grep "^  critic_lr:" ../../HARL/harl/configs/algos_cfgs/happo.yaml | awk '{print $2}')
critic_lr_new=$(grep "^  critic_lr:" ../../HARL/harl/configs/algos_cfgs/happo_safe.yaml | awk '{print $2}')
printf "%-25s | %-15s | %-15s\n" "Critic Learning Rate" "$critic_lr_old" "$critic_lr_new ⬇️"

grad_old=$(grep "^  max_grad_norm:" ../../HARL/harl/configs/algos_cfgs/happo.yaml | awk '{print $2}')
grad_new=$(grep "^  max_grad_norm:" ../../HARL/harl/configs/algos_cfgs/happo_safe.yaml | awk '{print $2}')
printf "%-25s | %-15s | %-15s\n" "Max Grad Norm" "$grad_old" "$grad_new ⚠️"

adv_norm_new=$(grep "^  use_adv_normalize:" ../../HARL/harl/configs/algos_cfgs/happo_safe.yaml | awk '{print $2}')
printf "%-25s | %-15s | %-15s\n" "Advantage Normalize" "Not set" "$adv_norm_new ✅"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Current train.sh setting:"
current_algo=$(grep '^algo=' train.sh | head -1 | cut -d'"' -f2)
echo "  algo=\"$current_algo\""
echo ""

if [ "$current_algo" = "happo_safe" ]; then
    echo "✅ Currently using SAFE config"
elif [ "$current_algo" = "happo" ]; then
    echo "⚠️  Currently using ORIGINAL config (may be unstable for long runs)"
fi

echo ""
echo "To switch to safe config:"
echo "  sed -i 's/^algo=\"happo\"/algo=\"happo_safe\"/' train.sh"
echo ""
echo "To switch to original config:"
echo "  sed -i 's/^algo=\"happo_safe\"/algo=\"happo\"/' train.sh"
echo ""

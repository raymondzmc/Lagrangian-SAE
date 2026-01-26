#!/bin/bash

# Launch 7 ReLU SAE sweeps in parallel on GPUs 1-7

# Get the project root directory (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SPARSITY_COEFFS=(8 9 14 25 30 35 40)
CONFIGS=(
    "configs/gemma2-2b/sweep/relu_sweeps/65K_relu_coeff_8.yaml"
    "configs/gemma2-2b/sweep/relu_sweeps/65K_relu_coeff_9.yaml"
    "configs/gemma2-2b/sweep/relu_sweeps/65K_relu_coeff_14.yaml"
    "configs/gemma2-2b/sweep/relu_sweeps/65K_relu_coeff_25.yaml"
    "configs/gemma2-2b/sweep/relu_sweeps/65K_relu_coeff_30.yaml"
    "configs/gemma2-2b/sweep/relu_sweeps/65K_relu_coeff_35.yaml"
    "configs/gemma2-2b/sweep/relu_sweeps/65K_relu_coeff_40.yaml"
)

for i in {0..6}; do
    SESSION_NAME="relu_sweep_$i"
    GPU=$((i + 1))
    CONFIG="${CONFIGS[$i]}"
    
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_ROOT && conda activate sae && CUDA_VISIBLE_DEVICES=$GPU python run_sweep.py --base_config configs/gemma2-2b/gemma2-relu.yaml --sweep_config $CONFIG" Enter
    
    echo "Started $SESSION_NAME (sparsity=${SPARSITY_COEFFS[$i]}) on GPU $GPU"
done

echo "All sessions started. Use 'tmux ls' to list sessions."

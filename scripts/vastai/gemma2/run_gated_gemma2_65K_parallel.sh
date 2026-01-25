#!/bin/bash

# Launch 5 Gated SAE sweeps in parallel on GPUs 0-4

# Get the project root directory (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SPARSITY_COEFFS=(0.012 0.018 0.024 0.04 0.06)

for i in {0..4}; do
    SESSION_NAME="gated_sweep_$i"
    GPU=$i
    CONFIG="configs/gemma2-2b/sweep/gated_sweeps/65K_gated_sweep_${i}.yaml"
    
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_ROOT && conda activate sae && CUDA_VISIBLE_DEVICES=$GPU python run_sweep.py --base_config configs/gemma2-2b/gemma2-gated.yaml --sweep_config $CONFIG" Enter
    
    echo "Started $SESSION_NAME (sparsity=${SPARSITY_COEFFS[$i]}) on GPU $GPU"
done

echo "All sessions started. Use 'tmux ls' to list sessions."

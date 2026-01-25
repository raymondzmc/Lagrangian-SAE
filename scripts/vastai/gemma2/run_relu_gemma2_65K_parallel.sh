#!/bin/bash

# Launch 7 ReLU SAE sweeps in parallel on GPUs 0-6

SPARSITY_COEFFS=(10 15 20 50 100 150 200)

for i in {0..6}; do
    SESSION_NAME="relu_sweep_${SPARSITY_COEFFS[$i]}"
    GPU=$i
    CONFIG="configs/gemma2-2b/sweep/relu_sweeps/65K_relu_sweep_${i}.yaml"
    
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "cd /workspace/Lagrangian-SAE && conda activate sae && CUDA_VISIBLE_DEVICES=$GPU python run_sweep.py --base_config configs/gemma2-2b/gemma2-relu.yaml --sweep_config $CONFIG" Enter
    
    echo "Started $SESSION_NAME on GPU $GPU"
done

echo "All sessions started. Use 'tmux ls' to list sessions."

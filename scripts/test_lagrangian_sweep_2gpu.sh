#!/bin/bash
# Test script: Run Lagrangian sweep with only 2 GPUs using tmux

SESSION_NAME="lagrangian_sweep_test"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session (detached)
tmux new-session -d -s $SESSION_NAME -n "gpu0"

SLEEP_DELAY=30  # seconds between starting each process to avoid HuggingFace download conflicts

# Window 0: GPU 0
tmux send-keys -t $SESSION_NAME:0 "conda activate sae && CUDA_VISIBLE_DEVICES=0 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_16_0.yaml" C-m
echo "Started GPU 0 (lagrangian_l0_16_0), waiting ${SLEEP_DELAY}s before next..."
sleep $SLEEP_DELAY

# Window 1: GPU 1
tmux new-window -t $SESSION_NAME -n "gpu1"
tmux send-keys -t $SESSION_NAME:1 "conda activate sae && CUDA_VISIBLE_DEVICES=1 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_16_1.yaml" C-m

echo "Started tmux session '$SESSION_NAME' with 2 parallel test jobs"
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To switch windows: Ctrl+b then 0-1"
echo "To detach: Ctrl+b then d"
echo "To kill all: tmux kill-session -t $SESSION_NAME"


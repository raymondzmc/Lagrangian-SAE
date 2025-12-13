#!/bin/bash
# Run all Lagrangian sweep configs in parallel using tmux

SESSION_NAME="lagrangian_sweep"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session (detached)
tmux new-session -d -s $SESSION_NAME -n "gpu0"

SLEEP_DELAY=30  # seconds between starting each process to avoid HuggingFace download conflicts

# Window 0: GPU 0
tmux send-keys -t $SESSION_NAME:0 "conda activate sae && CUDA_VISIBLE_DEVICES=0 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_16_0.yaml" C-m
echo "Started GPU 0, waiting ${SLEEP_DELAY}s before next..."
sleep $SLEEP_DELAY

# Window 1: GPU 1
tmux new-window -t $SESSION_NAME -n "gpu1"
tmux send-keys -t $SESSION_NAME:1 "conda activate sae && CUDA_VISIBLE_DEVICES=1 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_16_1.yaml" C-m
echo "Started GPU 1, waiting ${SLEEP_DELAY}s before next..."
sleep $SLEEP_DELAY

# Window 2: GPU 2
tmux new-window -t $SESSION_NAME -n "gpu2"
tmux send-keys -t $SESSION_NAME:2 "conda activate sae && CUDA_VISIBLE_DEVICES=2 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_32_0.yaml" C-m
echo "Started GPU 2, waiting ${SLEEP_DELAY}s before next..."
sleep $SLEEP_DELAY

# Window 3: GPU 3
tmux new-window -t $SESSION_NAME -n "gpu3"
tmux send-keys -t $SESSION_NAME:3 "conda activate sae && CUDA_VISIBLE_DEVICES=3 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_32_1.yaml" C-m
echo "Started GPU 3, waiting ${SLEEP_DELAY}s before next..."
sleep $SLEEP_DELAY

# Window 4: GPU 4
tmux new-window -t $SESSION_NAME -n "gpu4"
tmux send-keys -t $SESSION_NAME:4 "conda activate sae && CUDA_VISIBLE_DEVICES=4 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_64_0.yaml" C-m
echo "Started GPU 4, waiting ${SLEEP_DELAY}s before next..."
sleep $SLEEP_DELAY

# Window 5: GPU 5
tmux new-window -t $SESSION_NAME -n "gpu5"
tmux send-keys -t $SESSION_NAME:5 "conda activate sae && CUDA_VISIBLE_DEVICES=5 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_64_1.yaml" C-m
echo "Started GPU 5, waiting ${SLEEP_DELAY}s before next..."
sleep $SLEEP_DELAY

# Window 6: GPU 6
tmux new-window -t $SESSION_NAME -n "gpu6"
tmux send-keys -t $SESSION_NAME:6 "conda activate sae && CUDA_VISIBLE_DEVICES=6 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_128_0.yaml" C-m
echo "Started GPU 6, waiting ${SLEEP_DELAY}s before next..."
sleep $SLEEP_DELAY

# Window 7: GPU 7
tmux new-window -t $SESSION_NAME -n "gpu7"
tmux send-keys -t $SESSION_NAME:7 "conda activate sae && CUDA_VISIBLE_DEVICES=7 python run_sweep.py --base_config configs/gpt2/gpt2-lagrangian.yaml --sweep_config configs/gpt2/sweep/lagrangian_l0_128_1.yaml" C-m

echo "Started tmux session '$SESSION_NAME' with 8 parallel jobs"
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To switch windows: Ctrl+b then 0-7"
echo "To detach: Ctrl+b then d"
echo "To kill all: tmux kill-session -t $SESSION_NAME"


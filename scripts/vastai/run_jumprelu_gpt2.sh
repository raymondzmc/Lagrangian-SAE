#!/bin/bash

# Run 4 JumpReLU sweeps in parallel using tmux panes
# Usage: ./run_jumprelu_gpt2.sh

SESSION="jumprelu_sweep"
WORKDIR="/workspace/Lagrangian-SAE"
CONDA_CMD="conda activate sae"

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new tmux session with first command
tmux new-session -d -s $SESSION -c $WORKDIR

# Pane 0: target_l0_16
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=4 python run_sweep.py --base_config configs/gpt2/gpt2-jumprelu.yaml --sweep_config configs/gpt2/sweep/target_l0_16.yaml" C-m

# Split horizontally for pane 1: target_l0_32
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=5 python run_sweep.py --base_config configs/gpt2/gpt2-jumprelu.yaml --sweep_config configs/gpt2/sweep/target_l0_32.yaml" C-m

# Split pane 0 vertically for pane 2: target_l0_64
tmux select-pane -t $SESSION:0.0
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=6 python run_sweep.py --base_config configs/gpt2/gpt2-jumprelu.yaml --sweep_config configs/gpt2/sweep/target_l0_64.yaml" C-m

# Split pane 1 vertically for pane 3: target_l0_128
tmux select-pane -t $SESSION:0.1
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=7 python run_sweep.py --base_config configs/gpt2/gpt2-jumprelu.yaml --sweep_config configs/gpt2/sweep/target_l0_128.yaml" C-m

# Add pane titles (requires tmux 2.6+)
tmux select-pane -t $SESSION:0.0 -T "L0=16 (GPU4)"
tmux select-pane -t $SESSION:0.1 -T "L0=32 (GPU5)"
tmux select-pane -t $SESSION:0.2 -T "L0=64 (GPU6)"
tmux select-pane -t $SESSION:0.3 -T "L0=128 (GPU7)"

# Enable pane titles display
tmux set-option -t $SESSION pane-border-status top

echo "Started tmux session '$SESSION' with 4 parallel sweeps"
echo "Attach with: tmux attach -t $SESSION"
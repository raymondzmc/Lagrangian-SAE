#!/bin/bash

# Run ReLU and Gated experiments in parallel using tmux split panes
# Usage: ./run_relu_gated.sh

SESSION="gemma2_relu_gated"
WORKDIR="/workspace/Lagrangian-SAE"
CONDA_CMD="conda activate sae"

echo "=========================================="
echo "Gemma2-2B 50M ReLU & Gated Experiments"
echo "=========================================="
echo "GPU 0: ReLU 16K sweep"
echo "GPU 1: ReLU 65K sweep"
echo "GPU 2: Gated 16K sweep"
echo "GPU 3: Gated 65K sweep"
echo "=========================================="

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new tmux session with first command
tmux new-session -d -s $SESSION -c $WORKDIR

# Pane 0: ReLU 16K (GPU 0)
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=0 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-relu.yaml --sweep_config configs/gemma2-2b-50M/sweep/16K_relu_sweep.yaml" C-m

# Split horizontally for pane 1: ReLU 65K (GPU 1)
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=1 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-relu.yaml --sweep_config configs/gemma2-2b-50M/sweep/65K_relu_sweep.yaml" C-m

# Split pane 0 vertically for pane 2: Gated 16K (GPU 2)
tmux split-window -v -t $SESSION:0.0
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=2 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-gated.yaml --sweep_config configs/gemma2-2b-50M/sweep/16K_gated_sweep.yaml" C-m

# Split pane 1 vertically for pane 3: Gated 65K (GPU 3)
tmux split-window -v -t $SESSION:0.1
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=3 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-gated.yaml --sweep_config configs/gemma2-2b-50M/sweep/65K_gated_sweep.yaml" C-m

# Add pane titles (requires tmux 2.6+)
tmux select-pane -t $SESSION:0.0 -T "ReLU 16K (GPU0)"
tmux select-pane -t $SESSION:0.1 -T "ReLU 65K (GPU1)"
tmux select-pane -t $SESSION:0.2 -T "Gated 16K (GPU2)"
tmux select-pane -t $SESSION:0.3 -T "Gated 65K (GPU3)"

# Enable pane titles display
tmux set-option -t $SESSION pane-border-status top

# Use tiled layout for better visibility with 4 panes
tmux select-layout -t $SESSION tiled

echo ""
echo "Started tmux session '$SESSION' with 4 parallel experiments"
echo "Attach with: tmux attach -t $SESSION"

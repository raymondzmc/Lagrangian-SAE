#!/bin/bash

# Run Matryoshka BatchTopK vs Lagrangian comparison using tmux split panes
# Usage: ./run_matryoshka_comparison.sh

SESSION="matryoshka_comparison"
WORKDIR="/workspace/Lagrangian-SAE"
CONDA_CMD="conda activate sae"

echo "=========================================="
echo "Matryoshka SAE Comparison Experiment"
echo "=========================================="
echo "- BatchTopK (K=64) on GPU 2"
echo "- Lagrangian (target_l0=64) on GPU 3"
echo "- Training samples: 100,000"
echo "- Wandb project: gpt2-small-matryoshka"
echo "=========================================="

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new tmux session with first command
tmux new-session -d -s $SESSION -c $WORKDIR

# Pane 0: Matryoshka BatchTopK (K=64)
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=2 python3 run.py --config_path configs/gpt2/experiments/matryoshka_batchtopk_k64.yaml" C-m

# Split horizontally for pane 1: Matryoshka Lagrangian (target_l0=64)
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=3 python3 run.py --config_path configs/gpt2/experiments/matryoshka_lagrangian_k64.yaml" C-m

# Add pane titles (requires tmux 2.6+)
tmux select-pane -t $SESSION:0.0 -T "BatchTopK K=64 (GPU2)"
tmux select-pane -t $SESSION:0.1 -T "Lagrangian L0=64 (GPU3)"

# Enable pane titles display
tmux set-option -t $SESSION pane-border-status top

echo ""
echo "Started tmux session '$SESSION' with 2 parallel experiments"
echo "Attach with: tmux attach -t $SESSION"

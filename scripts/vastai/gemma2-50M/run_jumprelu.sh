#!/bin/bash

# Run JumpReLU experiments in parallel using tmux split panes
# Usage: ./run_jumprelu.sh

SESSION="gemma2_jumprelu_new_sparsity_coeff_10_warmup_0"
WORKDIR="/workspace/Lagrangian-SAE"
CONDA_CMD="conda activate sae"

echo "=========================================="
echo "Gemma2-2B 50M JumpReLU Experiments"
echo "=========================================="
echo "GPU 6: JumpReLU 16K sweep"
echo "GPU 7: JumpReLU 65K sweep"
echo "=========================================="

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new tmux session with first command
tmux new-session -d -s $SESSION -c $WORKDIR

# Pane 0: JumpReLU 16K (GPU 6)
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=2 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-jumprelu.yaml --sweep_config configs/gemma2-2b-50M/sweep/16K_l0_sweep.yaml" C-m

# Split horizontally for pane 1: JumpReLU 65K (GPU 7)
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=3 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-jumprelu.yaml --sweep_config configs/gemma2-2b-50M/sweep/65K_l0_sweep.yaml" C-m

# Add pane titles (requires tmux 2.6+)
tmux select-pane -t $SESSION:0.0 -T "JumpReLU 16K (GPU6)"
tmux select-pane -t $SESSION:0.1 -T "JumpReLU 65K (GPU7)"

# Enable pane titles display
tmux set-option -t $SESSION pane-border-status top

echo ""
echo "Started tmux session '$SESSION' with 2 parallel experiments"
echo "Attach with: tmux attach -t $SESSION"

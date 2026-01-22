#!/bin/bash

# Run 6 baseline experiments in parallel using tmux split panes
# Usage: ./run_baselines.sh

SESSION="gemma2_baselines"
WORKDIR="/workspace/Lagrangian-SAE"
CONDA_CMD="conda activate sae"

echo "=========================================="
echo "Gemma2-2B 50M Baseline Experiments"
echo "=========================================="
echo "GPU 0: TopK 16K sweep"
echo "GPU 1: TopK 65K sweep"
echo "GPU 2: BatchTopK 16K sweep"
echo "GPU 3: BatchTopK 65K sweep"
echo "GPU 4: Matryoshka 16K sweep"
echo "GPU 5: Matryoshka 65K sweep"
echo "=========================================="

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new tmux session with first command
tmux new-session -d -s $SESSION -c $WORKDIR

# Pane 0: TopK 16K (GPU 0)
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=0 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-topk.yaml --sweep_config configs/gemma2-2b-50M/sweep/16K_topk_sweep.yaml" C-m

# Split horizontally for pane 1: TopK 65K (GPU 1)
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=1 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-topk.yaml --sweep_config configs/gemma2-2b-50M/sweep/65K_topk_sweep.yaml" C-m

# Split pane 0 vertically for pane 2: BatchTopK 16K (GPU 2)
tmux split-window -v -t $SESSION:0.0
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=2 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-batchtopk.yaml --sweep_config configs/gemma2-2b-50M/sweep/16K_topk_sweep.yaml" C-m

# Split pane 1 vertically for pane 3: BatchTopK 65K (GPU 3)
tmux split-window -v -t $SESSION:0.1
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=3 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-batchtopk.yaml --sweep_config configs/gemma2-2b-50M/sweep/65K_topk_sweep.yaml" C-m

# Split pane 2 vertically for pane 4: Matryoshka 16K (GPU 4)
tmux split-window -v -t $SESSION:0.2
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=4 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-matryoshka.yaml --sweep_config configs/gemma2-2b-50M/sweep/16K_topk_sweep.yaml" C-m

# Split pane 3 vertically for pane 5: Matryoshka 65K (GPU 5)
tmux split-window -v -t $SESSION:0.3
tmux send-keys -t $SESSION "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=5 python run_sweep.py --base_config configs/gemma2-2b-50M/gemma2-matryoshka.yaml --sweep_config configs/gemma2-2b-50M/sweep/65K_topk_sweep.yaml" C-m

# Add pane titles (requires tmux 2.6+)
tmux select-pane -t $SESSION:0.0 -T "TopK 16K (GPU0)"
tmux select-pane -t $SESSION:0.1 -T "TopK 65K (GPU1)"
tmux select-pane -t $SESSION:0.2 -T "BatchTopK 16K (GPU2)"
tmux select-pane -t $SESSION:0.3 -T "BatchTopK 65K (GPU3)"
tmux select-pane -t $SESSION:0.4 -T "Matryoshka 16K (GPU4)"
tmux select-pane -t $SESSION:0.5 -T "Matryoshka 65K (GPU5)"

# Enable pane titles display
tmux set-option -t $SESSION pane-border-status top

# Use tiled layout for better visibility with 6 panes
tmux select-layout -t $SESSION tiled

echo ""
echo "Started tmux session '$SESSION' with 6 parallel experiments"
echo "Attach with: tmux attach -t $SESSION"

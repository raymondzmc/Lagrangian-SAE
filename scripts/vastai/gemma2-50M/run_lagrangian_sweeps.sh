#!/bin/bash

# Run 6 Lagrangian sweep experiments in parallel using tmux windows
# Usage: ./run_lagrangian_sweeps.sh

SESSION="gemma2_lagrangian"
WORKDIR="/workspace/Lagrangian-SAE"
CONDA_CMD="conda activate sae"
BASE_CONFIG="configs/gemma2-2b-50M/gemma2-lagrangian.yaml"
SWEEP_DIR="configs/gemma2-2b-50M/sweep/lagrangian_sweeps"

echo "=========================================="
echo "Gemma2-2B 50M Lagrangian Sweep Experiments"
echo "=========================================="
echo "GPU 2: Lagrangian 16K alpha_max=100"
echo "GPU 3: Lagrangian 16K alpha_max=50"
echo "GPU 4: Lagrangian 16K alpha_max=10"
echo "GPU 5: Lagrangian 65K alpha_max=100"
echo "GPU 6: Lagrangian 65K alpha_max=50"
echo "GPU 7: Lagrangian 65K alpha_max=10"
echo "=========================================="

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new tmux session with first window
tmux new-session -d -s $SESSION -n "16K_a100_GPU2" -c $WORKDIR

# Window 0: Lagrangian 16K alpha_max=100 (GPU 2)
tmux send-keys -t $SESSION:0 "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=2 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/16K_l0_sweep_0.yaml" C-m

# Window 1: Lagrangian 16K alpha_max=50 (GPU 3)
tmux new-window -t $SESSION -n "16K_a50_GPU3"
tmux send-keys -t $SESSION:1 "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=3 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/16K_l0_sweep_1.yaml" C-m

# Window 2: Lagrangian 16K alpha_max=10 (GPU 4)
tmux new-window -t $SESSION -n "16K_a10_GPU4"
tmux send-keys -t $SESSION:2 "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=4 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/16K_l0_sweep_2.yaml" C-m

# Window 3: Lagrangian 65K alpha_max=100 (GPU 5)
tmux new-window -t $SESSION -n "65K_a100_GPU5"
tmux send-keys -t $SESSION:3 "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=5 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_0.yaml" C-m

# Window 4: Lagrangian 65K alpha_max=50 (GPU 6)
tmux new-window -t $SESSION -n "65K_a50_GPU6"
tmux send-keys -t $SESSION:4 "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=6 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_1.yaml" C-m

# Window 5: Lagrangian 65K alpha_max=10 (GPU 7)
tmux new-window -t $SESSION -n "65K_a10_GPU7"
tmux send-keys -t $SESSION:5 "cd $WORKDIR && $CONDA_CMD && CUDA_VISIBLE_DEVICES=7 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_2.yaml" C-m

# Select first window
tmux select-window -t $SESSION:0

echo ""
echo "Started tmux session '$SESSION' with 6 parallel experiments (in separate windows)"
echo "Attach with: tmux attach -t $SESSION"
echo "Switch windows with: Ctrl-b n (next) or Ctrl-b p (previous) or Ctrl-b <number>"

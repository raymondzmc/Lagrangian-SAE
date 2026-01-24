#!/bin/bash

# Run 4 Lagrangian sweep experiments in parallel using tmux windows
# Usage: ./run_lagrangian_sweeps.sh

SESSION="gemma2_lagrangian"
WORKDIR="/workspace/Lagrangian-SAE"
VENV_CMD="source /venv/sae/bin/activate"
BASE_CONFIG="configs/gemma2-2b/gemma2-lagrangian.yaml"
SWEEP_DIR="configs/gemma2-2b/sweep/lagrangian_sweeps"

echo "=========================================="
echo "Gemma2-2B Lagrangian Sweep Experiments"
echo "=========================================="
echo "GPU 3: 65K alpha_max=5  equality=false (sweep_0)"
echo "GPU 4: 65K alpha_max=10 equality=false (sweep_1)"
echo "GPU 5: 65K alpha_max=5  equality=true  (sweep_2)"
echo "GPU 6: 65K alpha_max=10 equality=true  (sweep_3)"
echo "=========================================="

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new tmux session with first window
tmux new-session -d -s $SESSION -n "65K_a5_GPU3" -c $WORKDIR

# Window 0: Lagrangian 65K alpha_max=5 equality=false (GPU 3)
tmux send-keys -t $SESSION:0 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=3 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_0.yaml" C-m

# Window 1: Lagrangian 65K alpha_max=10 equality=false (GPU 4)
tmux new-window -t $SESSION -n "65K_a10_GPU4"
tmux send-keys -t $SESSION:1 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=4 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_1.yaml" C-m

# Window 2: Lagrangian 65K alpha_max=5 equality=true (GPU 5)
tmux new-window -t $SESSION -n "65K_a5eq_GPU5"
tmux send-keys -t $SESSION:2 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=5 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_2.yaml" C-m

# Window 3: Lagrangian 65K alpha_max=10 equality=true (GPU 6)
tmux new-window -t $SESSION -n "65K_a10eq_GPU6"
tmux send-keys -t $SESSION:3 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=6 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_3.yaml" C-m

# Select first window
tmux select-window -t $SESSION:0

echo ""
echo "Started tmux session '$SESSION' with 4 parallel experiments (in separate windows)"
echo "Attach with: tmux attach -t $SESSION"
echo "Switch windows with: Ctrl-b n (next) or Ctrl-b p (previous) or Ctrl-b <number>"

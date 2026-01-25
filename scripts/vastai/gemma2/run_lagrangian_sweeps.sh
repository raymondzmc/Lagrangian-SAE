#!/bin/bash

# Run 8 Lagrangian sweep experiments in parallel using separate tmux sessions
# Usage: ./run_lagrangian_sweeps.sh

SESSION_PREFIX="gemma2_lagrangian"
WORKDIR="/workspace/Lagrangian-SAE"
VENV_CMD="source /venv/sae/bin/activate"
BASE_CONFIG="configs/gemma2-2b/gemma2-lagrangian.yaml"
SWEEP_DIR="configs/gemma2-2b/sweep/lagrangian_sweeps"

echo "=========================================="
echo "Gemma2-2B Lagrangian Sweep Experiments"
echo "=========================================="
echo "GPU 0: 65K alpha_max=5  equality=false (sweep_0)"
echo "GPU 1: 65K alpha_max=10 equality=false (sweep_1)"
echo "GPU 2: 65K alpha_max=5  equality=true  (sweep_2)"
echo "GPU 3: 65K alpha_max=10 equality=true  (sweep_3)"
echo "GPU 4: 65K alpha_max=5  alpha_min=1 equality=true  (sweep_4)"
echo "GPU 5: 65K alpha_max=10 alpha_min=1 equality=true  (sweep_5)"
echo "GPU 6: 65K alpha_max=1  equality=true  (sweep_6)"
echo "GPU 7: 65K alpha_max=1  equality=false (sweep_7)"
echo "=========================================="

# Kill existing sessions if they exist
for i in {0..7}; do
    tmux kill-session -t ${SESSION_PREFIX}_${i} 2>/dev/null
done

# Session 0: Lagrangian 65K alpha_max=5 equality=false (GPU 0)
tmux new-session -d -s ${SESSION_PREFIX}_0 -c $WORKDIR
tmux send-keys -t ${SESSION_PREFIX}_0 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=0 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_0.yaml" C-m

# Session 1: Lagrangian 65K alpha_max=10 equality=false (GPU 1)
tmux new-session -d -s ${SESSION_PREFIX}_1 -c $WORKDIR
tmux send-keys -t ${SESSION_PREFIX}_1 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=1 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_1.yaml" C-m

# Session 2: Lagrangian 65K alpha_max=5 equality=true (GPU 2)
tmux new-session -d -s ${SESSION_PREFIX}_2 -c $WORKDIR
tmux send-keys -t ${SESSION_PREFIX}_2 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=2 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_2.yaml" C-m

# Session 3: Lagrangian 65K alpha_max=10 equality=true (GPU 3)
tmux new-session -d -s ${SESSION_PREFIX}_3 -c $WORKDIR
tmux send-keys -t ${SESSION_PREFIX}_3 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=3 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_3.yaml" C-m

# Session 4: Lagrangian 65K alpha_max=5 alpha_min=1 equality=true (GPU 4)
tmux new-session -d -s ${SESSION_PREFIX}_4 -c $WORKDIR
tmux send-keys -t ${SESSION_PREFIX}_4 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=4 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_4.yaml" C-m

# Session 5: Lagrangian 65K alpha_max=10 alpha_min=1 equality=true (GPU 5)
tmux new-session -d -s ${SESSION_PREFIX}_5 -c $WORKDIR
tmux send-keys -t ${SESSION_PREFIX}_5 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=5 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_5.yaml" C-m

# Session 6: Lagrangian 65K alpha_max=1 equality=true (GPU 6)
tmux new-session -d -s ${SESSION_PREFIX}_6 -c $WORKDIR
tmux send-keys -t ${SESSION_PREFIX}_6 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=6 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_6.yaml" C-m

# Session 7: Lagrangian 65K alpha_max=1 equality=false (GPU 7)
tmux new-session -d -s ${SESSION_PREFIX}_7 -c $WORKDIR
tmux send-keys -t ${SESSION_PREFIX}_7 "cd $WORKDIR && $VENV_CMD && CUDA_VISIBLE_DEVICES=7 python run_sweep.py --base_config $BASE_CONFIG --sweep_config $SWEEP_DIR/65K_l0_sweep_7.yaml" C-m

echo ""
echo "Started 8 separate tmux sessions with parallel experiments:"
echo "  ${SESSION_PREFIX}_0 through ${SESSION_PREFIX}_7"
echo ""
echo "List sessions: tmux ls"
echo "Attach to session N: tmux attach -t ${SESSION_PREFIX}_N"

#!/bin/bash
# Run SAEBench evaluations for ALL gemma2-2b-65K SAEs on GPUs 4-7
# Each GPU runs its assigned SAEs sequentially in a tmux session
# Total: 28 runs distributed evenly across 4 GPUs (7 each)

# Configuration
PROJECT="gemma2-2b-65K"
EVAL_TYPES="core autointerp scr tpp ravel"
OUTPUT_BASE="./saebench_results"

# Kill existing sessions if they exist
tmux kill-session -t saebench_gpu4 2>/dev/null
tmux kill-session -t saebench_gpu5 2>/dev/null
tmux kill-session -t saebench_gpu6 2>/dev/null
tmux kill-session -t saebench_gpu7 2>/dev/null

echo "========================================"
echo "SAEBench Evaluation - All 65K SAEs"
echo "========================================"
echo "Project: $PROJECT"
echo "Eval types: $EVAL_TYPES"
echo "GPUs: 4, 5, 6, 7"
echo "Total runs: 28 (7 per GPU)"
echo ""

# Helper function to generate run command
run_cmd() {
    local gpu=$1
    local run_name=$2
    local output_name=$3
    echo "CUDA_VISIBLE_DEVICES=$gpu python3 run_saebench.py --project $PROJECT --run_name \"$run_name\" --eval_types $EVAL_TYPES --device cuda:0 --output_path ${OUTPUT_BASE}/${output_name} --force_rerun"
}

# ============================================================
# GPU 4: TopK (4) + BatchTopK (3) = 7 runs
# ============================================================
echo "Starting GPU 4: TopK + BatchTopK (7 runs)..."
tmux new-session -d -s saebench_gpu4
tmux send-keys -t saebench_gpu4 "cd /workspace/Lagrangian-SAE && \
$(run_cmd 4 "topk_k_32_n_dict_components_65536" "topk_k32") && \
$(run_cmd 4 "topk_k_64_n_dict_components_65536" "topk_k64") && \
$(run_cmd 4 "topk_k_128_n_dict_components_65536" "topk_k128") && \
$(run_cmd 4 "topk_k_256_n_dict_components_65536" "topk_k256") && \
$(run_cmd 4 "batch_topk_k_32_n_dict_components_65536" "batch_topk_k32") && \
$(run_cmd 4 "batch_topk_k_64_n_dict_components_65536" "batch_topk_k64") && \
$(run_cmd 4 "batch_topk_k_128_n_dict_components_65536" "batch_topk_k128") && \
echo 'GPU 4 COMPLETE'" Enter

# ============================================================
# GPU 5: BatchTopK (1) + JumpReLU (4) + Matryoshka (2) = 7 runs
# ============================================================
echo "Starting GPU 5: BatchTopK + JumpReLU + Matryoshka (7 runs)..."
tmux new-session -d -s saebench_gpu5
tmux send-keys -t saebench_gpu5 "cd /workspace/Lagrangian-SAE && \
$(run_cmd 5 "batch_topk_k_256_n_dict_components_65536" "batch_topk_k256") && \
$(run_cmd 5 "jumprelu_target_l0_32_n_dict_components_65536" "jumprelu_l0_32") && \
$(run_cmd 5 "jumprelu_target_l0_64_n_dict_components_65536" "jumprelu_l0_64") && \
$(run_cmd 5 "jumprelu_target_l0_128_n_dict_components_65536" "jumprelu_l0_128") && \
$(run_cmd 5 "jumprelu_target_l0_256_n_dict_components_65536" "jumprelu_l0_256") && \
$(run_cmd 5 "matryoshka_k_32_n_dict_components_65536" "matryoshka_k32") && \
$(run_cmd 5 "matryoshka_k_64_n_dict_components_65536" "matryoshka_k64") && \
echo 'GPU 5 COMPLETE'" Enter

# ============================================================
# GPU 6: Matryoshka (2) + Lagrangian l0=32 (3) + l0=64 (2) = 7 runs
# ============================================================
echo "Starting GPU 6: Matryoshka + Lagrangian l0=32,64 (7 runs)..."
tmux new-session -d -s saebench_gpu6
tmux send-keys -t saebench_gpu6 "cd /workspace/Lagrangian-SAE && \
$(run_cmd 6 "matryoshka_k_128_n_dict_components_65536" "matryoshka_k128") && \
$(run_cmd 6 "matryoshka_k_256_n_dict_components_65536" "matryoshka_k256") && \
$(run_cmd 6 "lagrangian_target_l0_32_alpha_max_1_n_dict_components_65536" "lagrangian_l0_32_alpha1") && \
$(run_cmd 6 "lagrangian_target_l0_32_alpha_max_5_n_dict_components_65536" "lagrangian_l0_32_alpha5") && \
$(run_cmd 6 "lagrangian_target_l0_32_alpha_max_10_n_dict_components_65536" "lagrangian_l0_32_alpha10") && \
$(run_cmd 6 "lagrangian_target_l0_64_alpha_max_1_n_dict_components_65536" "lagrangian_l0_64_alpha1") && \
$(run_cmd 6 "lagrangian_target_l0_64_alpha_max_5_n_dict_components_65536" "lagrangian_l0_64_alpha5") && \
echo 'GPU 6 COMPLETE'" Enter

# ============================================================
# GPU 7: Lagrangian l0=64 (1) + l0=128 (3) + l0=256 (3) = 7 runs
# ============================================================
echo "Starting GPU 7: Lagrangian l0=64,128,256 (7 runs)..."
tmux new-session -d -s saebench_gpu7
tmux send-keys -t saebench_gpu7 "cd /workspace/Lagrangian-SAE && \
$(run_cmd 7 "lagrangian_target_l0_64_alpha_max_10_n_dict_components_65536" "lagrangian_l0_64_alpha10") && \
$(run_cmd 7 "lagrangian_target_l0_128_alpha_max_1_n_dict_components_65536" "lagrangian_l0_128_alpha1") && \
$(run_cmd 7 "lagrangian_target_l0_128_alpha_max_5_n_dict_components_65536" "lagrangian_l0_128_alpha5") && \
$(run_cmd 7 "lagrangian_target_l0_128_alpha_max_10_n_dict_components_65536" "lagrangian_l0_128_alpha10") && \
$(run_cmd 7 "lagrangian_target_l0_256_alpha_max_1_n_dict_components_65536" "lagrangian_l0_256_alpha1") && \
$(run_cmd 7 "lagrangian_target_l0_256_alpha_max_5_n_dict_components_65536" "lagrangian_l0_256_alpha5") && \
$(run_cmd 7 "lagrangian_target_l0_256_alpha_max_10_n_dict_components_65536" "lagrangian_l0_256_alpha10") && \
echo 'GPU 7 COMPLETE'" Enter

echo ""
echo "========================================"
echo "All sessions started!"
echo "========================================"
echo ""
echo "GPU Assignment (7 runs each):"
echo "  GPU 4: TopK (4) + BatchTopK (3)"
echo "  GPU 5: BatchTopK (1) + JumpReLU (4) + Matryoshka (2)"
echo "  GPU 6: Matryoshka (2) + Lagrangian l0=32 (3) + l0=64 (2)"
echo "  GPU 7: Lagrangian l0=64 (1) + l0=128 (3) + l0=256 (3)"
echo ""
echo "Monitor commands:"
echo "  tmux ls                    # List all sessions"
echo "  tmux attach -t saebench_gpu4   # Attach to GPU 4 session"
echo "  tmux attach -t saebench_gpu5   # Attach to GPU 5 session"
echo "  tmux attach -t saebench_gpu6   # Attach to GPU 6 session"
echo "  tmux attach -t saebench_gpu7   # Attach to GPU 7 session"
echo ""
echo "Detach from session: Ctrl+B, then D"
echo "Kill all sessions: tmux kill-session -t saebench_gpu4 && tmux kill-session -t saebench_gpu5 && tmux kill-session -t saebench_gpu6 && tmux kill-session -t saebench_gpu7"

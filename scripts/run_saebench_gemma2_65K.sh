#!/bin/bash
# Run SAEBench evaluations for gemma2-2b-65K SAEs on GPUs 3-6
# Each evaluation runs in a separate tmux session

# Configuration
PROJECT="gemma2-2b-65K"
EVAL_TYPES="core autointerp scr tpp ravel sparse_probing_sae_probes"
OUTPUT_BASE="./saebench_results"

# Kill existing sessions if they exist
tmux kill-session -t saebench_topk 2>/dev/null
tmux kill-session -t saebench_batch_topk 2>/dev/null
tmux kill-session -t saebench_matryoshka 2>/dev/null
tmux kill-session -t saebench_jumprelu 2>/dev/null

echo "Starting SAEBench evaluations..."
echo "Project: $PROJECT"
echo "Eval types: $EVAL_TYPES"
echo ""

# Session 1: GPU 3 - TopK SAE
echo "Starting TopK evaluation on GPU 3..."
tmux new-session -d -s saebench_topk
tmux send-keys -t saebench_topk "cd /workspace/Lagrangian-SAE && CUDA_VISIBLE_DEVICES=3 python3 run_saebench.py \
  --project $PROJECT \
  --run_name \"topk_k_32_n_dict_components_65536\" \
  --eval_types $EVAL_TYPES \
  --device cuda:0 \
  --output_path ${OUTPUT_BASE}/topk_k32" Enter

# Session 2: GPU 4 - BatchTopK SAE
echo "Starting BatchTopK evaluation on GPU 4..."
tmux new-session -d -s saebench_batch_topk
tmux send-keys -t saebench_batch_topk "cd /workspace/Lagrangian-SAE && CUDA_VISIBLE_DEVICES=4 python3 run_saebench.py \
  --project $PROJECT \
  --run_name \"batch_topk_k_32_n_dict_components_65536\" \
  --eval_types $EVAL_TYPES \
  --device cuda:0 \
  --output_path ${OUTPUT_BASE}/batch_topk_k32" Enter

# Session 3: GPU 5 - Matryoshka SAE
echo "Starting Matryoshka evaluation on GPU 5..."
tmux new-session -d -s saebench_matryoshka
tmux send-keys -t saebench_matryoshka "cd /workspace/Lagrangian-SAE && CUDA_VISIBLE_DEVICES=5 python3 run_saebench.py \
  --project $PROJECT \
  --run_name \"matryoshka_k_32_n_dict_components_65536\" \
  --eval_types $EVAL_TYPES \
  --device cuda:0 \
  --output_path ${OUTPUT_BASE}/matryoshka_k32" Enter

# Session 4: GPU 6 - JumpReLU SAE (force_rerun to verify decode() fix)
echo "Starting JumpReLU evaluation on GPU 6..."
tmux new-session -d -s saebench_jumprelu
tmux send-keys -t saebench_jumprelu "cd /workspace/Lagrangian-SAE && CUDA_VISIBLE_DEVICES=6 python3 run_saebench.py \
  --project $PROJECT \
  --run_name \"jumprelu_target_l0_32_n_dict_components_65536\" \
  --eval_types $EVAL_TYPES \
  --device cuda:0 \
  --output_path ${OUTPUT_BASE}/jumprelu_l0_32 \
  --force_rerun" Enter

echo ""
echo "All sessions started!"
echo ""
echo "Monitor commands:"
echo "  tmux ls                        # List all sessions"
echo "  tmux attach -t saebench_topk   # Attach to TopK session"
echo "  tmux attach -t saebench_batch_topk   # Attach to BatchTopK session"
echo "  tmux attach -t saebench_matryoshka   # Attach to Matryoshka session"
echo "  tmux attach -t saebench_jumprelu     # Attach to JumpReLU session"
echo ""
echo "Detach from session: Ctrl+B, then D"
echo "Kill all sessions: tmux kill-server"

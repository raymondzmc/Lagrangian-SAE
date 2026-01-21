conda activate sae

CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-batchtopk.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_topk_sweep.yaml
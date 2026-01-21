conda activate sae

CUDA_VISIBLE_DEVICES=2 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-batchtopk.yaml \
--sweep_config configs/gemma2-2b/sweep/16K_topk_sweep.yaml
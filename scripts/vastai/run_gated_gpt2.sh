conda activate sae

CUDA_VISIBLE_DEVICES=1 python run_sweep.py \
--base_config configs/gpt2/gpt2-gated.yaml \
--sweep_config configs/gpt2/sweep/gated_sweep.yaml
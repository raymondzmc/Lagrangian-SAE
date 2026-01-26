conda activate sae

CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gpt2/gpt2-lagrangian.yaml \
--sweep_config configs/gpt2/sweep/lagrangian/lagrangian_l0_alpha_5.yaml
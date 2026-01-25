conda activate sae
CUDA_VISIBLE_DEVICES=6 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-matryoshka-lagrangian.yaml \
--sweep_config configs/gemma2-2b/sweep/lagrangian_sweeps/65K_l0_sweep_1.yaml
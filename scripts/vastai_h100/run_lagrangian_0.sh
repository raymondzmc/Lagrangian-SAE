conda activate sae
CUDA_VISIBLE_DEVICES=0 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-lagrangian.yaml \
--sweep_config configs/gemma2-2b/sweep/lagrangian_sweeps/65K_l0_sweep_0.yaml

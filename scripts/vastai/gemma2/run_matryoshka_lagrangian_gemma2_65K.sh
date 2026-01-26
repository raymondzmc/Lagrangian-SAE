source /venv/sae/bin/activate

CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-matryoshka-lagrangian.yaml \
--sweep_config configs/gemma2-2b/sweep/lagrangian_sweeps/65K_l0_sweep_5.yaml

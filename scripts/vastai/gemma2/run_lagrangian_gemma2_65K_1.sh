source /venv/sae/bin/activate

CUDA_VISIBLE_DEVICES=4 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-lagrangian.yaml \
--sweep_config configs/gemma2-2b/sweep/lagrangian_sweeps/65K_l0_sweep_1.yaml

source /venv/sae/bin/activate

CUDA_VISIBLE_DEVICES=1 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-matryoshka-lagrangian.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_l0_sweep.yaml

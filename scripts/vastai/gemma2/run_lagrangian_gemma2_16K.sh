source /venv/sae/bin/activate

CUDA_VISIBLE_DEVICES=6 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-lagrangian.yaml \
--sweep_config configs/gemma2-2b/sweep/16K_l0_sweep.yaml

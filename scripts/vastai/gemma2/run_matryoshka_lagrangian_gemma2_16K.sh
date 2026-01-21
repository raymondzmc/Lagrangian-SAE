source /venv/sae/bin/activate

CUDA_VISIBLE_DEVICES=0 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-matryoshka-lagrangian.yaml \
--sweep_config configs/gemma2-2b/sweep/16K_l0_sweep.yaml

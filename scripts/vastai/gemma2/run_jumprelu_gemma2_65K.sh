source /venv/sae/bin/activate

CUDA_VISIBLE_DEVICES=7 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-jumprelu.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_l0_sweep.yaml

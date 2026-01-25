source /venv/sae/bin/activate

CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-matryoshka.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_K_128.yaml

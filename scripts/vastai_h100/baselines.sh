CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-batchtopk.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_K_64.yaml

CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-batchtopk.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_K_128.yaml

CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-batchtopk.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_K_256.yaml

CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-topk.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_K_256.yaml

CUDA_VISIBLE_DEVICES=3 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-jumprelu.yaml \
--sweep_config configs/gemma2-2b/sweep/65K_K_256.yaml
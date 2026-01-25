conda activate sae

CUDA_VISIBLE_DEVICES=1 python run_sweep.py \
--base_config configs/gemma2-2b/gemma2-relu.yaml \
--sweep_config configs/gemma2-2b/sweep/relu_sweeps/65K_relu_coeff_13.yaml
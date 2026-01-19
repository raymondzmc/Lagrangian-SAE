conda activate sae

CUDA_VISIBLE_DEVICES=0 python run_sweep.py \
--base_config configs/gpt2/gpt2-relu.yaml \
--sweep_config configs/gpt2/sweep/relu_sweep.yaml
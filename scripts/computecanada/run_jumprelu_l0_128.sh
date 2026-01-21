#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100
#SBATCH --job-name=jumprelu_l0_128

module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load cuda/12.6

source ~/venvs/sae/bin/activate


python run_sweep.py \
--base_config configs/gpt2/gpt2-jumprelu.yaml \
--sweep_config configs/gpt2/sweep/target_l0_128.yaml
module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load cuda/12.6

# python -m venv ~/venvs/sae
source ~/venvs/sae/bin/activate
pip install --upgrade pip


pip install torch --index-url https://download.pytorch.org/whl/cu126


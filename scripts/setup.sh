conda create -n sae python=3.12 -y
conda activate sae
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
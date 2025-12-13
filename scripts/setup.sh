conda create -n sae python=3.12
conda activate sae
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
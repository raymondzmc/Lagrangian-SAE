salloc \
  --account=def-carenini \
  --time=03:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:h100:1 \
  --constraint=h100


# For Trillium (Change partition to compute for longer runs)
salloc \
  --account=def-carenini \
  --partition=debug \
  --time=01:00:00 \
  --cpus-per-task=8 \
  --gpus-per-node=1
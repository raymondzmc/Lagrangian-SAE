# Lagrangian-SAE

This repository contains implementations of various Sparse Autoencoder (SAE) architectures, including the Lagrangian SAE. It supports training and evaluation on TransformerLens models.

## Usage

### Running a Single Experiment

To run a single training experiment, use `run.py` with a configuration YAML file:

```bash
python run.py --config_path configs/your_config.yaml --device cuda:0
```

**Arguments:**
- `--config_path`: Path to the YAML configuration file.
- `--device`: Device to use (e.g., `cuda:0`, `cpu`). Defaults to `cuda:0` if available.

### Running a Hyperparameter Sweep

To run a sweep of experiments, use `run_sweep.py`. This script takes a base configuration and a sweep configuration to generate and execute multiple experiments.

```bash
python run_sweep.py \
    --base_config configs/base_config.yaml \
    --sweep_config configs/sweep_config.yaml \
    --device cuda:0
```

**Arguments:**
- `--base_config`: Path to the base configuration file containing common settings.
- `--sweep_config`: Path to the sweep configuration file defining parameter grids.
- `--device`: Device to use for the experiments.
- `--dry_run`: Print generated configurations without running them.
- `--show_devices`: List available compute devices.
- `--limit`: Limit the number of experiments to run (useful for testing).

## Outputs and Logging

### Local Storage
By default, the following are saved locally in the `output` directory (configurable via `save_dir` in config):
- **Model Checkpoints**: Saved as `.pt` files (e.g., `samples_100000.pt`) containing the SAE state dict.
- **Configuration**: A copy of the configuration used for the run.

### Weights & Biases (Wandb)
If `wandb_project` is specified in the configuration, the following metrics are logged to Wandb:

**Training Metrics:**
- `loss`: Total loss.
- `mse_loss`: Mean Squared Error reconstruction loss.
- `sparsity_loss`: Sparsity penalty (if applicable).
- `aux_loss`: Auxiliary loss for dead feature mitigation (if used).
- `l0`: Average L0 norm (number of active features).
- `explained_variance`: Explained variance of the reconstruction.
- `alive_dict_components`: Number of active dictionary components.

**Artifacts:**
- The trained model checkpoints are also uploaded to Wandb as artifacts if configured.

## Configuration Structure

### Base Config
Defines the standard settings for the model, data, and training loop.
```yaml
wandb_project: "my-project"
saes:
  sae_type: "lagrangian"
  # ... other SAE params
data:
  dataset_name: "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
  # ... data params
```

### Sweep Config
Defines parameters to vary. Lists are treated as grids to search over.
```yaml
lr: [1e-3, 3e-4]
saes:
    sparsity_coeff: [0.1, 1.0, 10.0]
```

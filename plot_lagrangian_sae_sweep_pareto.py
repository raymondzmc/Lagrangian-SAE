#!/usr/bin/env python3
"""
Plot Pareto curves for Lagrangian SAE hyperparameter sweeps.

Plots separate Pareto curves for each combination of:
- l0_ema_momentum: [0.9, 0.99]
- alpha_max: [0.5, 1.0]
- bandwidth: [0.1, 0.01, 0.001]
- rho_quadratic: [0.1, 0.001, 0]

Total: 2 × 2 × 3 × 3 = 36 configurations
Each configuration has runs for K = 16, 32, 64, 128 (target_l0 values)
"""

import re
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from itertools import product

from settings import settings
from utils.io import load_metrics_from_wandb


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


# Define the hyperparameter grid
HYPERPARAMETER_GRID = {
    'l0_ema_momentum': [0.9, 0.99],
    'alpha_max': [0.5, 1.0],
    'bandwidth': [0.1, 0.01, 0.001],
    'rho_quadratic': [0.1, 0.001, 0],
}

# Valid K values (target_l0)
VALID_K_VALUES = [16, 32, 64, 128]


def generate_all_configurations() -> List[Dict[str, float]]:
    """
    Generate all 36 hyperparameter configurations.
    
    Returns:
        List of dictionaries, each containing one configuration
    """
    param_names = list(HYPERPARAMETER_GRID.keys())
    param_values = list(HYPERPARAMETER_GRID.values())
    
    configurations = []
    for values in product(*param_values):
        config = dict(zip(param_names, values))
        configurations.append(config)
    
    return configurations


def format_param_value(param_name: str, value: float) -> str:
    """Format a parameter value for run name matching."""
    if value == 0:
        return "0"
    elif value < 0.01:
        return f"{value:.0e}"
    elif isinstance(value, float) and value == int(value):
        return str(int(value))
    else:
        return str(value)


def build_run_name_pattern(config: Dict[str, float], target_l0: int) -> str:
    """
    Build a regex pattern to match run names for a given configuration.
    
    The run_sweep.py generates run names like:
    lagrangian_target_l0_16_l0_ema_momentum_0.9_alpha_max_0.5_bandwidth_0.1_rho_quadratic_0.1
    """
    parts = [f"lagrangian_target_l0_{target_l0}"]
    for param_name, value in config.items():
        formatted_value = format_param_value(param_name, value)
        parts.append(f"{param_name}_{formatted_value}")
    
    # Build a pattern that matches these parts in any order (after the lagrangian_target_l0_ prefix)
    # Since run_sweep.py uses consistent ordering, we can be more specific
    pattern = "_".join(parts)
    return pattern


def config_matches_run_name(run_name: str, config: Dict[str, float], target_l0: int) -> bool:
    """
    Check if a run name matches the given configuration and target_l0.
    
    Args:
        run_name: The Wandb run name
        config: Configuration dictionary
        target_l0: Target L0 value
    
    Returns:
        True if the run matches the configuration
    """
    import re
    
    # Check target_l0 - must be exact match (not substring like 16 matching 116)
    # Pattern: target_l0_{K}_ or target_l0_{K} at end
    target_pattern = f"target_l0_{target_l0}_"
    if target_pattern not in run_name:
        # Check if it's at the end of the run name
        if not run_name.endswith(f"target_l0_{target_l0}"):
            return False
    
    # Check each hyperparameter with exact matching
    for param_name, value in config.items():
        # Build pattern with word boundary after the value
        # The pattern should be: {param_name}_{value}_ or {param_name}_{value} at end of segment
        
        # Generate possible value formats
        if value == 0:
            value_patterns = ["0", "0.0"]
        elif value < 0.01:
            value_patterns = [
                f"{value:.0e}",  # Scientific notation like 1e-03
                f"{value}",
                f"{value:.3f}",
                "1e-03" if value == 0.001 else f"{value:.0e}",
            ]
        elif value == int(value):
            value_patterns = [str(int(value)), f"{value}", f"{value:.1f}"]
        else:
            value_patterns = [f"{value}", f"{value:.1f}", f"{value:.2f}"]
        
        # Check if any pattern matches exactly (followed by _ or end of param section)
        matched = False
        for val_pat in value_patterns:
            # Pattern should be followed by underscore (next param) or nothing before n_train_samples
            exact_pattern = f"{param_name}_{val_pat}_"
            if exact_pattern in run_name:
                matched = True
                break
            # Also check for end of parameters (before n_train_samples)
            if f"{param_name}_{val_pat}_n_train" in run_name:
                matched = True
                break
        
        if not matched:
            return False
    
    return True


def collect_sweep_metrics_data(
    project: str = None
) -> Tuple[Dict[Tuple, List[Dict]], List[str]]:
    """
    Collect metrics data for all Lagrangian SAE sweep configurations.
    
    Args:
        project: Wandb project name for sweep runs. If None, uses settings.wandb_entity/gpt2-small-sweep
    
    Returns:
        Tuple of (data_by_config, layers) where data_by_config is a dictionary
        mapping configuration tuples to lists of run data
    """
    if project is None:
        project = f"{settings.wandb_entity}/gpt2-small-sweep"
    
    print(f"Collecting metrics data from Wandb project: {project}")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs from the project
    print(f"Fetching runs from {project}...")
    try:
        runs = list(api.runs(project))
    except Exception as e:
        print(f"Error fetching runs: {e}")
        runs = []
    
    print(f"Found {len(runs)} total runs")
    
    # Generate all configurations
    configurations = generate_all_configurations()
    print(f"Looking for {len(configurations)} hyperparameter configurations")
    
    # Collect data by configuration
    # Key: tuple of (l0_ema_momentum, alpha_max, bandwidth, rho_quadratic)
    data_by_config: Dict[Tuple, List[Dict]] = {}
    all_layers = set()
    
    # First pass: collect runs by name to handle duplicates (keep latest)
    runs_by_name = {}
    for run in runs:
        name = run.name
        if name.startswith("lagrangian_"):
            if name not in runs_by_name:
                runs_by_name[name] = run
            else:
                # Keep the latest run
                existing_run = runs_by_name[name]
                if run.created_at > existing_run.created_at:
                    runs_by_name[name] = run
    
    print(f"After deduplication: {len(runs_by_name)} unique Lagrangian runs")
    
    # Process runs and match to configurations
    for run in runs_by_name.values():
        name = run.name
        
        # Try to match this run to a configuration
        matched_config = None
        matched_k = None
        
        for config in configurations:
            for k in VALID_K_VALUES:
                if config_matches_run_name(name, config, k):
                    matched_config = config
                    matched_k = k
                    break
            if matched_config is not None:
                break
        
        if matched_config is None:
            continue
        
        # Create config key
        config_key = (
            matched_config['l0_ema_momentum'],
            matched_config['alpha_max'],
            matched_config['bandwidth'],
            matched_config['rho_quadratic'],
        )
        
        # Load metrics for this run
        print(f"  Loading metrics for {name} (k={matched_k}, config={config_key})")
        metrics = load_metrics_from_wandb(run.id, project)
        
        if metrics:
            run_data = {
                'run_name': name,
                'run_id': run.id,
                'target_l0': matched_k,
                'config': matched_config,
                'layers': {}
            }
            
            # Process each layer
            for layer_name, layer_metrics in metrics.items():
                all_layers.add(layer_name)
                run_data['layers'][layer_name] = {
                    'l0': layer_metrics['sparsity_l0'],
                    'mse': layer_metrics['mse'],
                    'explained_variance': layer_metrics['explained_variance'],
                    'alive_dict_components': layer_metrics.get('alive_dict_components', 0),
                    'alive_dict_proportion': layer_metrics.get('alive_dict_components_proportion', 0)
                }
            
            if config_key not in data_by_config:
                data_by_config[config_key] = []
            data_by_config[config_key].append(run_data)
    
    # Sort each configuration's runs by L0
    for config_key in data_by_config:
        data_by_config[config_key] = sorted(
            data_by_config[config_key],
            key=lambda x: np.mean([m['l0'] for m in x['layers'].values()])
        )
    
    print(f"\nCollected data summary:")
    print(f"  Configurations with data: {len(data_by_config)}/{len(configurations)}")
    for config_key, runs in data_by_config.items():
        k_values = sorted(set(r['target_l0'] for r in runs))
        print(f"    {config_key}: {len(runs)} runs (K={k_values})")
    print(f"  Layers found: {sorted(all_layers)}")
    
    return data_by_config, sorted(all_layers)


def collect_baseline_metrics_data(
    project: str = None
) -> Tuple[Dict[str, List[Dict]], List[str]]:
    """
    Collect metrics data for TopK and BatchTopK baseline runs.
    
    Args:
        project: Wandb project name for baseline runs. If None, uses settings.wandb_entity/gpt2-small
    
    Returns:
        Tuple of (baseline_data, layers) where baseline_data is a dictionary
        with 'topk' and 'batch_topk' keys containing lists of run data
    """
    if project is None:
        project = f"{settings.wandb_entity}/gpt2-small"
    
    print(f"\nCollecting baseline metrics data from Wandb project: {project}")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs from the project
    print(f"Fetching runs from {project}...")
    try:
        runs = list(api.runs(project))
    except Exception as e:
        print(f"Error fetching runs: {e}")
        runs = []
    
    print(f"Found {len(runs)} total runs")
    
    # Collect data by SAE type
    baseline_data = {
        'topk': [],         # runs with "topk_k_{K}"
        'batch_topk': [],   # runs with "batch_topk_k_{K}"
    }
    
    all_layers = set()
    
    # First pass: collect runs by name to handle duplicates (keep latest)
    runs_by_name = {}
    for run in runs:
        name = run.name
        topk_match = re.match(r'^topk_k_(\d+)$', name)
        batch_topk_match = re.match(r'^batch_topk_k_(\d+)$', name)
        
        if topk_match or batch_topk_match:
            if name not in runs_by_name:
                runs_by_name[name] = run
            else:
                # Keep the latest run
                existing_run = runs_by_name[name]
                if run.created_at > existing_run.created_at:
                    runs_by_name[name] = run
    
    print(f"After deduplication: {len(runs_by_name)} unique baseline runs")
    
    # Process runs
    for run in runs_by_name.values():
        name = run.name
        
        # Determine SAE type based on run name patterns
        sae_type = None
        k_value = None
        
        topk_match = re.match(r'^topk_k_(\d+)$', name)
        batch_topk_match = re.match(r'^batch_topk_k_(\d+)$', name)
        
        if topk_match:
            k_value = int(topk_match.group(1))
            if k_value in VALID_K_VALUES:
                sae_type = 'topk'
        elif batch_topk_match:
            k_value = int(batch_topk_match.group(1))
            if k_value in VALID_K_VALUES:
                sae_type = 'batch_topk'
        
        if sae_type is None:
            continue
        
        # Load metrics for this run
        print(f"  Loading metrics for {name} (k={k_value}, type={sae_type})")
        metrics = load_metrics_from_wandb(run.id, project)
        
        if metrics:
            run_data = {
                'run_name': name,
                'run_id': run.id,
                'k_value': k_value,
                'layers': {}
            }
            
            # Process each layer
            for layer_name, layer_metrics in metrics.items():
                all_layers.add(layer_name)
                run_data['layers'][layer_name] = {
                    'l0': layer_metrics['sparsity_l0'],
                    'mse': layer_metrics['mse'],
                    'explained_variance': layer_metrics['explained_variance'],
                    'alive_dict_components': layer_metrics.get('alive_dict_components', 0),
                    'alive_dict_proportion': layer_metrics.get('alive_dict_components_proportion', 0)
                }
            
            baseline_data[sae_type].append(run_data)
    
    # Sort by K value
    for sae_type in baseline_data:
        baseline_data[sae_type] = sorted(
            baseline_data[sae_type],
            key=lambda x: x['k_value']
        )
    
    print(f"\nBaseline data summary:")
    for sae_type, runs in baseline_data.items():
        k_values = sorted(set(r['k_value'] for r in runs))
        print(f"  {sae_type}: {len(runs)} runs (K={k_values})")
    print(f"  Layers found: {sorted(all_layers)}")
    
    return baseline_data, sorted(all_layers)


def find_pareto_frontier(x_values: np.ndarray, y_values: np.ndarray,
                         minimize_x: bool = True, minimize_y: bool = True) -> np.ndarray:
    """
    Find the Pareto frontier for 2D data.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values
        minimize_x: If True, prefer smaller x values
        minimize_y: If True, prefer smaller y values
    
    Returns:
        Boolean array indicating which points are on the Pareto frontier
    """
    n_points = len(x_values)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if minimize_x and minimize_y:
                    if x_values[j] <= x_values[i] and y_values[j] <= y_values[i]:
                        if x_values[j] < x_values[i] or y_values[j] < y_values[i]:
                            is_pareto[i] = False
                            break
                elif minimize_x and not minimize_y:
                    if x_values[j] <= x_values[i] and y_values[j] >= y_values[i]:
                        if x_values[j] < x_values[i] or y_values[j] > y_values[i]:
                            is_pareto[i] = False
                            break
    
    return is_pareto


def get_config_display_name(config_key: Tuple) -> str:
    """Create a human-readable display name for a configuration."""
    l0_ema, alpha_max, bandwidth, rho = config_key
    
    # Format bandwidth and rho nicely
    if bandwidth < 0.01:
        bw_str = f"{bandwidth:.0e}"
    else:
        bw_str = f"{bandwidth}"
    
    if rho == 0:
        rho_str = "0"
    elif rho < 0.01:
        rho_str = f"{rho:.0e}"
    else:
        rho_str = f"{rho}"
    
    return f"ema={l0_ema}, α_max={alpha_max}, bw={bw_str}, ρ={rho_str}"


def plot_sweep_pareto_curves(
    data_by_config: Dict[Tuple, List[Dict]],
    layers: List[str],
    output_dir: Path = Path("plots/sweep"),
    max_l0: float = 140.0,
    min_l0: float = 10.0
):
    """
    Create Pareto curve plots for each hyperparameter configuration.
    
    Args:
        data_by_config: Dictionary mapping config tuples to run data
        layers: List of layer names
        output_dir: Output directory for plots
        max_l0: Maximum L0 threshold for filtering
        min_l0: Minimum L0 threshold for filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating Pareto plots with L0 filtering: [{min_l0}, {max_l0}]")
    
    # Color palette for different target_l0 values
    k_colors = {
        16: '#1f77b4',   # Blue
        32: '#2ca02c',   # Green
        64: '#ff7f0e',   # Orange
        128: '#d62728', # Red
    }
    
    # Create plots for each layer
    for layer_name in layers:
        print(f"\nProcessing layer: {layer_name}")
        
        # Create a figure with subplots for each configuration
        # Arrange in a grid based on hyperparameter values
        n_configs = len(data_by_config)
        if n_configs == 0:
            print(f"  No data available for layer {layer_name}")
            continue
        
        # For 36 configs: 6 rows x 6 cols or grouped by parameter
        # Let's group by (l0_ema_momentum, alpha_max) for rows and (bandwidth, rho) for columns
        ema_values = sorted(set(k[0] for k in data_by_config.keys()))
        alpha_values = sorted(set(k[1] for k in data_by_config.keys()))
        bw_values = sorted(set(k[2] for k in data_by_config.keys()), reverse=True)
        rho_values = sorted(set(k[3] for k in data_by_config.keys()), reverse=True)
        
        n_rows = len(ema_values) * len(alpha_values)  # 2 * 2 = 4
        n_cols = len(bw_values) * len(rho_values)     # 3 * 3 = 9
        
        # Create figure - MSE vs L0
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
        
        row_idx = 0
        for ema in ema_values:
            for alpha in alpha_values:
                col_idx = 0
                for bw in bw_values:
                    for rho in rho_values:
                        config_key = (ema, alpha, bw, rho)
                        ax = axes[row_idx, col_idx]
                        
                        if config_key not in data_by_config:
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=10, color='gray')
                            ax.set_title(get_config_display_name(config_key), fontsize=8)
                            col_idx += 1
                            continue
                        
                        runs = data_by_config[config_key]
                        
                        # Collect data for this configuration
                        all_l0 = []
                        all_mse = []
                        all_k = []
                        
                        for run_data in runs:
                            if layer_name not in run_data['layers']:
                                continue
                            
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            k = run_data['target_l0']
                            
                            # Apply filtering
                            if min_l0 <= l0 <= max_l0:
                                all_l0.append(l0)
                                all_mse.append(mse)
                                all_k.append(k)
                        
                        if not all_l0:
                            ax.text(0.5, 0.5, 'No data\n(filtered)', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=10, color='gray')
                            ax.set_title(get_config_display_name(config_key), fontsize=8)
                            col_idx += 1
                            continue
                        
                        all_l0 = np.array(all_l0)
                        all_mse = np.array(all_mse)
                        all_k = np.array(all_k)
                        
                        # Plot points for each K value
                        for k in VALID_K_VALUES:
                            mask = all_k == k
                            if np.any(mask):
                                ax.scatter(all_l0[mask], all_mse[mask],
                                          color=k_colors[k], marker='o',
                                          alpha=0.8, s=60, label=f'K={k}')
                                
                                # Add K labels
                                for x, y in zip(all_l0[mask], all_mse[mask]):
                                    ax.annotate(f'{k}', (x, y),
                                               xytext=(2, 2), textcoords='offset points',
                                               fontsize=7, alpha=0.6, color=k_colors[k])
                        
                        # Find and plot Pareto frontier
                        is_pareto = find_pareto_frontier(all_l0, all_mse, 
                                                        minimize_x=True, minimize_y=True)
                        if np.any(is_pareto):
                            pareto_l0 = all_l0[is_pareto]
                            pareto_mse = all_mse[is_pareto]
                            sort_idx = np.argsort(pareto_l0)
                            ax.plot(pareto_l0[sort_idx], pareto_mse[sort_idx],
                                   color='black', linewidth=1.5, alpha=0.6, linestyle='--')
                        
                        ax.set_title(get_config_display_name(config_key), fontsize=8)
                        ax.tick_params(axis='both', labelsize=8)
                        ax.grid(True, alpha=0.3)
                        
                        col_idx += 1
                row_idx += 1
        
        # Add common labels and legend
        fig.text(0.5, 0.02, 'L0 Sparsity', ha='center', fontsize=14)
        fig.text(0.02, 0.5, 'MSE', va='center', rotation='vertical', fontsize=14)
        
        # Add legend to the first subplot
        handles = [plt.scatter([], [], color=k_colors[k], marker='o', s=60, label=f'K={k}')
                   for k in VALID_K_VALUES]
        axes[0, 0].legend(handles=handles, loc='upper right', fontsize=8)
        
        layer_display_name = layer_name.replace('.', '_')
        plt.suptitle(f'MSE vs L0 - Lagrangian SAE Sweep - {layer_name}', fontsize=16, y=0.995)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.99])
        
        # Save figure
        output_path = output_dir / f"lagrangian_sweep_mse_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"  Saved MSE plot to: {output_path}")
        plt.close()
        
        # Create figure - Explained Variance vs L0
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
        
        row_idx = 0
        for ema in ema_values:
            for alpha in alpha_values:
                col_idx = 0
                for bw in bw_values:
                    for rho in rho_values:
                        config_key = (ema, alpha, bw, rho)
                        ax = axes[row_idx, col_idx]
                        
                        if config_key not in data_by_config:
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=10, color='gray')
                            ax.set_title(get_config_display_name(config_key), fontsize=8)
                            col_idx += 1
                            continue
                        
                        runs = data_by_config[config_key]
                        
                        # Collect data for this configuration
                        all_l0 = []
                        all_ev = []
                        all_k = []
                        
                        for run_data in runs:
                            if layer_name not in run_data['layers']:
                                continue
                            
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            ev = run_data['layers'][layer_name]['explained_variance']
                            k = run_data['target_l0']
                            
                            # Apply filtering
                            if min_l0 <= l0 <= max_l0:
                                all_l0.append(l0)
                                all_ev.append(ev)
                                all_k.append(k)
                        
                        if not all_l0:
                            ax.text(0.5, 0.5, 'No data\n(filtered)', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=10, color='gray')
                            ax.set_title(get_config_display_name(config_key), fontsize=8)
                            col_idx += 1
                            continue
                        
                        all_l0 = np.array(all_l0)
                        all_ev = np.array(all_ev)
                        all_k = np.array(all_k)
                        
                        # Plot points for each K value
                        for k in VALID_K_VALUES:
                            mask = all_k == k
                            if np.any(mask):
                                ax.scatter(all_l0[mask], all_ev[mask],
                                          color=k_colors[k], marker='o',
                                          alpha=0.8, s=60, label=f'K={k}')
                                
                                # Add K labels
                                for x, y in zip(all_l0[mask], all_ev[mask]):
                                    ax.annotate(f'{k}', (x, y),
                                               xytext=(2, 2), textcoords='offset points',
                                               fontsize=7, alpha=0.6, color=k_colors[k])
                        
                        # Find and plot Pareto frontier (maximize EV)
                        is_pareto = find_pareto_frontier(all_l0, -all_ev,  # Negate for max
                                                        minimize_x=True, minimize_y=True)
                        if np.any(is_pareto):
                            pareto_l0 = all_l0[is_pareto]
                            pareto_ev = all_ev[is_pareto]
                            sort_idx = np.argsort(pareto_l0)
                            ax.plot(pareto_l0[sort_idx], pareto_ev[sort_idx],
                                   color='black', linewidth=1.5, alpha=0.6, linestyle='--')
                        
                        ax.set_title(get_config_display_name(config_key), fontsize=8)
                        ax.tick_params(axis='both', labelsize=8)
                        ax.grid(True, alpha=0.3)
                        
                        col_idx += 1
                row_idx += 1
        
        # Add common labels and legend
        fig.text(0.5, 0.02, 'L0 Sparsity', ha='center', fontsize=14)
        fig.text(0.02, 0.5, 'Explained Variance', va='center', rotation='vertical', fontsize=14)
        
        # Add legend to the first subplot
        handles = [plt.scatter([], [], color=k_colors[k], marker='o', s=60, label=f'K={k}')
                   for k in VALID_K_VALUES]
        axes[0, 0].legend(handles=handles, loc='lower right', fontsize=8)
        
        plt.suptitle(f'Explained Variance vs L0 - Lagrangian SAE Sweep - {layer_name}', fontsize=16, y=0.995)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.99])
        
        # Save figure
        output_path = output_dir / f"lagrangian_sweep_ev_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"  Saved EV plot to: {output_path}")
        plt.close()


def plot_sweep_summary(
    data_by_config: Dict[Tuple, List[Dict]],
    layers: List[str],
    output_dir: Path = Path("plots/sweep"),
    max_l0: float = 140.0,
    min_l0: float = 10.0
):
    """
    Create summary plots showing best configurations.
    
    Args:
        data_by_config: Dictionary mapping config tuples to run data
        layers: List of layer names
        output_dir: Output directory for plots
        max_l0: Maximum L0 threshold for filtering
        min_l0: Minimum L0 threshold for filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating summary plots...")
    
    # For each layer, find the best configuration for each target L0
    for layer_name in layers:
        print(f"\nProcessing layer: {layer_name}")
        
        # Collect best MSE for each config and target_l0
        results = []
        
        for config_key, runs in data_by_config.items():
            for run_data in runs:
                if layer_name not in run_data['layers']:
                    continue
                
                l0 = run_data['layers'][layer_name]['l0']
                mse = run_data['layers'][layer_name]['mse']
                ev = run_data['layers'][layer_name]['explained_variance']
                k = run_data['target_l0']
                
                if min_l0 <= l0 <= max_l0:
                    results.append({
                        'config': config_key,
                        'target_l0': k,
                        'l0': l0,
                        'mse': mse,
                        'ev': ev,
                    })
        
        if not results:
            print(f"  No data available")
            continue
        
        # Create summary heatmaps for each target_l0
        # Heatmap: bandwidth vs rho, separate for each (ema, alpha) combination
        for k in VALID_K_VALUES:
            k_results = [r for r in results if r['target_l0'] == k]
            if not k_results:
                continue
            
            # Get unique values
            bw_values = sorted(set(HYPERPARAMETER_GRID['bandwidth']), reverse=True)
            rho_values = sorted(set(HYPERPARAMETER_GRID['rho_quadratic']), reverse=True)
            ema_values = sorted(set(HYPERPARAMETER_GRID['l0_ema_momentum']))
            alpha_values = sorted(set(HYPERPARAMETER_GRID['alpha_max']))
            
            fig, axes = plt.subplots(len(ema_values), len(alpha_values), 
                                    figsize=(5*len(alpha_values), 4*len(ema_values)))
            if len(ema_values) == 1:
                axes = axes.reshape(1, -1)
            if len(alpha_values) == 1:
                axes = axes.reshape(-1, 1)
            
            for i, ema in enumerate(ema_values):
                for j, alpha in enumerate(alpha_values):
                    ax = axes[i, j]
                    
                    # Create MSE matrix
                    mse_matrix = np.full((len(bw_values), len(rho_values)), np.nan)
                    
                    for r in k_results:
                        config = r['config']
                        if config[0] == ema and config[1] == alpha:
                            bw_idx = bw_values.index(config[2])
                            rho_idx = rho_values.index(config[3])
                            mse_matrix[bw_idx, rho_idx] = r['mse']
                    
                    # Plot heatmap
                    im = ax.imshow(mse_matrix, cmap='RdYlGn_r', aspect='auto')
                    
                    # Add annotations
                    for bi in range(len(bw_values)):
                        for ri in range(len(rho_values)):
                            if not np.isnan(mse_matrix[bi, ri]):
                                ax.text(ri, bi, f'{mse_matrix[bi, ri]:.4f}',
                                       ha='center', va='center', fontsize=8)
                    
                    ax.set_xticks(range(len(rho_values)))
                    ax.set_yticks(range(len(bw_values)))
                    
                    # Format tick labels
                    rho_labels = ['0' if r == 0 else (f'{r:.0e}' if r < 0.01 else str(r)) for r in rho_values]
                    bw_labels = [f'{b:.0e}' if b < 0.01 else str(b) for b in bw_values]
                    
                    ax.set_xticklabels(rho_labels, fontsize=9)
                    ax.set_yticklabels(bw_labels, fontsize=9)
                    ax.set_xlabel('rho_quadratic', fontsize=10)
                    ax.set_ylabel('bandwidth', fontsize=10)
                    ax.set_title(f'ema={ema}, α_max={alpha}', fontsize=11)
                    
                    plt.colorbar(im, ax=ax, label='MSE')
            
            layer_display_name = layer_name.replace('.', '_')
            plt.suptitle(f'MSE Heatmap - K={k} - {layer_name}', fontsize=14)
            plt.tight_layout()
            
            output_path = output_dir / f"lagrangian_sweep_heatmap_k{k}_{layer_display_name}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=200)
            print(f"  Saved heatmap for K={k} to: {output_path}")
            plt.close()


def print_sweep_summary(
    data_by_config: Dict[Tuple, List[Dict]],
    layers: List[str],
    max_l0: float = 140.0,
    min_l0: float = 10.0
):
    """Print a summary of the best configurations."""
    print("\n" + "=" * 100)
    print("LAGRANGIAN SAE SWEEP SUMMARY")
    print("=" * 100)
    
    for layer_name in layers:
        print(f"\n{'='*100}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*100}")
        
        # Collect all results for this layer
        all_results = []
        
        for config_key, runs in data_by_config.items():
            for run_data in runs:
                if layer_name not in run_data['layers']:
                    continue
                
                l0 = run_data['layers'][layer_name]['l0']
                mse = run_data['layers'][layer_name]['mse']
                ev = run_data['layers'][layer_name]['explained_variance']
                k = run_data['target_l0']
                
                if min_l0 <= l0 <= max_l0:
                    all_results.append({
                        'config': config_key,
                        'target_l0': k,
                        'l0': l0,
                        'mse': mse,
                        'ev': ev,
                    })
        
        if not all_results:
            print("  No data available")
            continue
        
        # Find best config for each target_l0
        for k in VALID_K_VALUES:
            k_results = [r for r in all_results if r['target_l0'] == k]
            if not k_results:
                continue
            
            # Best by MSE
            best_mse = min(k_results, key=lambda x: x['mse'])
            # Best by EV
            best_ev = max(k_results, key=lambda x: x['ev'])
            
            print(f"\n  Target L0 = {k}:")
            print(f"    Best MSE: {best_mse['mse']:.6f} (L0={best_mse['l0']:.1f})")
            print(f"      Config: {get_config_display_name(best_mse['config'])}")
            print(f"    Best EV:  {best_ev['ev']:.4f} (L0={best_ev['l0']:.1f})")
            print(f"      Config: {get_config_display_name(best_ev['config'])}")
    
    print("\n" + "=" * 100)


def plot_overlay_pareto_curves(
    data_by_config: Dict[Tuple, List[Dict]],
    layers: List[str],
    output_dir: Path = Path("plots/sweep"),
    max_l0: float = 140.0,
    min_l0: float = 10.0
):
    """
    Create overlay Pareto curve plots showing all configurations on the same plot.
    
    Uses color coding by (ema, alpha_max) groups, marker styles by rho_quadratic,
    and highlights the global Pareto frontier.
    
    Args:
        data_by_config: Dictionary mapping config tuples to run data
        layers: List of layer names
        output_dir: Output directory for plots
        max_l0: Maximum L0 threshold for filtering
        min_l0: Minimum L0 threshold for filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating overlay Pareto plots...")
    
    # Color schemes for (ema, alpha_max) combinations
    # Each combination gets a color family with varying intensities for bandwidth
    color_schemes = {
        (0.9, 0.5): {'base': 'Blues', 'colors': ['#c6dbef', '#6baed6', '#2171b5']},      # Blues
        (0.9, 1.0): {'base': 'Greens', 'colors': ['#c7e9c0', '#74c476', '#238b45']},     # Greens  
        (0.99, 0.5): {'base': 'Oranges', 'colors': ['#fdd0a2', '#fd8d3c', '#d94801']},   # Oranges
        (0.99, 1.0): {'base': 'Reds', 'colors': ['#fcbba1', '#fb6a4a', '#cb181d']},      # Reds
    }
    
    # Marker styles for rho_quadratic values
    rho_markers = {
        0.1: 'o',     # Circle
        0.001: 's',   # Square
        0: '^',       # Triangle
    }
    
    # Bandwidth values for color intensity (sorted for indexing)
    bw_values = sorted(HYPERPARAMETER_GRID['bandwidth'], reverse=True)  # [0.1, 0.01, 0.001]
    
    for layer_name in layers:
        print(f"\nProcessing layer: {layer_name}")
        
        # Create figure with two subplots: MSE vs L0 and EV vs L0
        fig, (ax_mse, ax_ev) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Collect all data points for this layer
        all_points_mse = []  # (l0, mse, config_key, target_l0)
        all_points_ev = []   # (l0, ev, config_key, target_l0)
        
        for config_key, runs in data_by_config.items():
            for run_data in runs:
                if layer_name not in run_data['layers']:
                    continue
                
                l0 = run_data['layers'][layer_name]['l0']
                mse = run_data['layers'][layer_name]['mse']
                ev = run_data['layers'][layer_name]['explained_variance']
                k = run_data['target_l0']
                
                if min_l0 <= l0 <= max_l0:
                    all_points_mse.append((l0, mse, config_key, k))
                    all_points_ev.append((l0, ev, config_key, k))
        
        if not all_points_mse:
            print(f"  No data available for layer {layer_name}")
            plt.close()
            continue
        
        # Plot each configuration
        legend_handles = []
        legend_labels = []
        
        for config_key in sorted(data_by_config.keys()):
            ema, alpha_max, bandwidth, rho = config_key
            
            # Get color based on (ema, alpha_max) and bandwidth
            scheme_key = (ema, alpha_max)
            if scheme_key not in color_schemes:
                continue
            
            bw_idx = bw_values.index(bandwidth)
            color = color_schemes[scheme_key]['colors'][bw_idx]
            
            # Get marker based on rho
            marker = rho_markers.get(rho, 'o')
            
            # Filter points for this config
            config_mse = [(p[0], p[1], p[3]) for p in all_points_mse if p[2] == config_key]
            config_ev = [(p[0], p[1], p[3]) for p in all_points_ev if p[2] == config_key]
            
            if not config_mse:
                continue
            
            # Plot MSE
            l0_vals = [p[0] for p in config_mse]
            mse_vals = [p[1] for p in config_mse]
            k_vals = [p[2] for p in config_mse]
            
            scatter = ax_mse.scatter(l0_vals, mse_vals, c=color, marker=marker, 
                                     s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Add K labels near points
            for x, y, k in zip(l0_vals, mse_vals, k_vals):
                ax_mse.annotate(f'{k}', (x, y), xytext=(3, 3), textcoords='offset points',
                              fontsize=6, alpha=0.5)
            
            # Plot EV
            l0_vals_ev = [p[0] for p in config_ev]
            ev_vals = [p[1] for p in config_ev]
            k_vals_ev = [p[2] for p in config_ev]
            
            ax_ev.scatter(l0_vals_ev, ev_vals, c=color, marker=marker,
                         s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            for x, y, k in zip(l0_vals_ev, ev_vals, k_vals_ev):
                ax_ev.annotate(f'{k}', (x, y), xytext=(3, 3), textcoords='offset points',
                              fontsize=6, alpha=0.5)
        
        # Find and plot global Pareto frontier for MSE
        l0_array = np.array([p[0] for p in all_points_mse])
        mse_array = np.array([p[1] for p in all_points_mse])
        
        is_pareto_mse = find_pareto_frontier(l0_array, mse_array, minimize_x=True, minimize_y=True)
        if np.any(is_pareto_mse):
            pareto_l0 = l0_array[is_pareto_mse]
            pareto_mse = mse_array[is_pareto_mse]
            sort_idx = np.argsort(pareto_l0)
            ax_mse.plot(pareto_l0[sort_idx], pareto_mse[sort_idx], 
                       color='black', linewidth=2, linestyle='-', label='Pareto Frontier', zorder=10)
            ax_mse.scatter(pareto_l0, pareto_mse, c='black', marker='*', s=150, 
                          zorder=11, label='Pareto Points')
        
        # Find and plot global Pareto frontier for EV (maximize EV)
        ev_array = np.array([p[1] for p in all_points_ev])
        
        is_pareto_ev = find_pareto_frontier(l0_array, -ev_array, minimize_x=True, minimize_y=True)
        if np.any(is_pareto_ev):
            pareto_l0 = l0_array[is_pareto_ev]
            pareto_ev = ev_array[is_pareto_ev]
            sort_idx = np.argsort(pareto_l0)
            ax_ev.plot(pareto_l0[sort_idx], pareto_ev[sort_idx],
                      color='black', linewidth=2, linestyle='-', label='Pareto Frontier', zorder=10)
            ax_ev.scatter(pareto_l0, pareto_ev, c='black', marker='*', s=150,
                         zorder=11, label='Pareto Points')
        
        # Configure MSE plot
        ax_mse.set_xlabel('L0 Sparsity', fontsize=12)
        ax_mse.set_ylabel('MSE', fontsize=12)
        ax_mse.set_title(f'MSE vs L0 - {layer_name}', fontsize=14)
        ax_mse.grid(True, alpha=0.3)
        
        # Configure EV plot
        ax_ev.set_xlabel('L0 Sparsity', fontsize=12)
        ax_ev.set_ylabel('Explained Variance', fontsize=12)
        ax_ev.set_title(f'Explained Variance vs L0 - {layer_name}', fontsize=14)
        ax_ev.grid(True, alpha=0.3)
        
        # Create custom legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # (ema, alpha_max) color legend
        for (ema, alpha_max), scheme in sorted(color_schemes.items()):
            legend_elements.append(
                Patch(facecolor=scheme['colors'][1], edgecolor='white',
                     label=f'ema={ema}, α={alpha_max}')
            )
        
        legend_elements.append(Line2D([0], [0], color='none', label=''))  # Spacer
        
        # Marker legend for rho
        for rho, marker in sorted(rho_markers.items(), reverse=True):
            rho_str = '0' if rho == 0 else (f'{rho:.0e}' if rho < 0.01 else str(rho))
            legend_elements.append(
                Line2D([0], [0], marker=marker, color='gray', linestyle='None',
                      markersize=8, label=f'ρ={rho_str}')
            )
        
        legend_elements.append(Line2D([0], [0], color='none', label=''))  # Spacer
        
        # Bandwidth intensity note
        legend_elements.append(
            Patch(facecolor='white', edgecolor='gray',
                 label='Color intensity: bw')
        )
        legend_elements.append(
            Patch(facecolor='lightgray', edgecolor='gray',
                 label='(light→dark = 0.1→0.001)')
        )
        
        legend_elements.append(Line2D([0], [0], color='none', label=''))  # Spacer
        
        # Pareto frontier
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=2, linestyle='-',
                  marker='*', markersize=10, label='Pareto Frontier')
        )
        
        ax_mse.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                     framealpha=0.9, ncol=1)
        ax_ev.legend(handles=legend_elements, loc='lower right', fontsize=8,
                    framealpha=0.9, ncol=1)
        
        plt.tight_layout()
        
        # Save figure
        layer_display_name = layer_name.replace('.', '_')
        output_path = output_dir / f"lagrangian_sweep_overlay_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"  Saved overlay plot to: {output_path}")
        plt.close()


# Top 8 Lagrangian configurations by total wins (from comprehensive analysis)
TOP_LAGRANGIAN_CONFIGS = [
    (0.99, 0.5, 0.1, 0.001),    # 20 wins
    (0.9, 0.5, 0.1, 0),         # 10 wins
    (0.99, 0.5, 0.001, 0),      # 6 wins
    (0.9, 0.5, 0.1, 0.001),     # 5 wins
    (0.99, 1.0, 0.1, 0),        # 4 wins
    (0.99, 1.0, 0.1, 0.001),    # 4 wins
    (0.99, 1.0, 0.01, 0),       # 4 wins
    (0.99, 0.5, 0.1, 0.1),      # 3 wins
]


def plot_comparison_pareto_curves(
    lagrangian_data: Dict[Tuple, List[Dict]],
    baseline_data: Dict[str, List[Dict]],
    layers: List[str],
    top_n_configs: int = 8,
    output_dir: Path = Path("plots/sweep"),
    max_l0: float = 140.0,
    min_l0: float = 10.0
):
    """
    Create comparison plots: TopK and BatchTopK baselines vs top Lagrangian configs.
    
    Args:
        lagrangian_data: Dictionary mapping config tuples to run data
        baseline_data: Dictionary with 'topk' and 'batch_topk' keys
        layers: List of layer names
        top_n_configs: Number of top Lagrangian configs to include
        output_dir: Output directory for plots
        max_l0: Maximum L0 threshold for filtering
        min_l0: Minimum L0 threshold for filtering
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating baseline comparison plots...")
    print(f"  Using top {top_n_configs} Lagrangian configurations")
    
    # Get top N configurations
    top_configs = TOP_LAGRANGIAN_CONFIGS[:top_n_configs]
    
    # Color scheme
    baseline_colors = {
        'topk': '#1f77b4',       # Blue
        'batch_topk': '#2ca02c', # Green
    }
    baseline_markers = {
        'topk': 'o',             # Circle
        'batch_topk': 's',       # Square
    }
    baseline_labels = {
        'topk': 'TopK',
        'batch_topk': 'BatchTopK',
    }
    
    # Warm color gradient for Lagrangian configs (red to orange to yellow)
    lagrangian_cmap = plt.cm.YlOrRd
    lagrangian_colors = [lagrangian_cmap(0.9 - i * 0.08) for i in range(top_n_configs)]
    
    for layer_name in layers:
        print(f"\nProcessing layer: {layer_name}")
        
        # Create figure with two subplots: MSE vs L0 and EV vs L0
        fig, (ax_mse, ax_ev) = plt.subplots(1, 2, figsize=(20, 9))
        
        # ===== Plot Baselines =====
        for sae_type in ['topk', 'batch_topk']:
            runs = baseline_data.get(sae_type, [])
            if not runs:
                continue
            
            l0_values = []
            mse_values = []
            ev_values = []
            k_labels = []
            
            for run_data in runs:
                if layer_name not in run_data['layers']:
                    continue
                
                l0 = run_data['layers'][layer_name]['l0']
                mse = run_data['layers'][layer_name]['mse']
                ev = run_data['layers'][layer_name]['explained_variance']
                k = run_data['k_value']
                
                if min_l0 <= l0 <= max_l0:
                    l0_values.append(l0)
                    mse_values.append(mse)
                    ev_values.append(ev)
                    k_labels.append(k)
            
            if not l0_values:
                continue
            
            l0_values = np.array(l0_values)
            mse_values = np.array(mse_values)
            ev_values = np.array(ev_values)
            
            # Plot MSE
            ax_mse.scatter(l0_values, mse_values,
                          color=baseline_colors[sae_type],
                          marker=baseline_markers[sae_type],
                          s=200, alpha=0.9, edgecolors='black', linewidth=1,
                          label=baseline_labels[sae_type], zorder=5)
            
            # Add K labels
            for x, y, k in zip(l0_values, mse_values, k_labels):
                ax_mse.annotate(f'{k}', (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=10, alpha=0.7, color=baseline_colors[sae_type],
                               fontweight='bold')
            
            # Connect baseline points with line
            sort_idx = np.argsort(l0_values)
            ax_mse.plot(l0_values[sort_idx], mse_values[sort_idx],
                       color=baseline_colors[sae_type], linewidth=2, alpha=0.6)
            
            # Plot EV
            ax_ev.scatter(l0_values, ev_values,
                         color=baseline_colors[sae_type],
                         marker=baseline_markers[sae_type],
                         s=200, alpha=0.9, edgecolors='black', linewidth=1,
                         label=baseline_labels[sae_type], zorder=5)
            
            for x, y, k in zip(l0_values, ev_values, k_labels):
                ax_ev.annotate(f'{k}', (x, y), xytext=(5, 5), textcoords='offset points',
                              fontsize=10, alpha=0.7, color=baseline_colors[sae_type],
                              fontweight='bold')
            
            ax_ev.plot(l0_values[sort_idx], ev_values[sort_idx],
                      color=baseline_colors[sae_type], linewidth=2, alpha=0.6)
        
        # ===== Plot Top Lagrangian Configs =====
        all_lagrangian_mse_points = []
        all_lagrangian_ev_points = []
        
        for config_idx, config_key in enumerate(top_configs):
            if config_key not in lagrangian_data:
                print(f"  Warning: Config {config_key} not found in data")
                continue
            
            runs = lagrangian_data[config_key]
            color = lagrangian_colors[config_idx]
            
            l0_values = []
            mse_values = []
            ev_values = []
            k_labels = []
            
            for run_data in runs:
                if layer_name not in run_data['layers']:
                    continue
                
                l0 = run_data['layers'][layer_name]['l0']
                mse = run_data['layers'][layer_name]['mse']
                ev = run_data['layers'][layer_name]['explained_variance']
                k = run_data['target_l0']
                
                if min_l0 <= l0 <= max_l0:
                    l0_values.append(l0)
                    mse_values.append(mse)
                    ev_values.append(ev)
                    k_labels.append(k)
                    all_lagrangian_mse_points.append((l0, mse))
                    all_lagrangian_ev_points.append((l0, ev))
            
            if not l0_values:
                continue
            
            l0_values = np.array(l0_values)
            mse_values = np.array(mse_values)
            ev_values = np.array(ev_values)
            
            # Create short label for legend
            ema, alpha, bw, rho = config_key
            rho_str = '0' if rho == 0 else (f'{rho:.0e}' if rho < 0.01 else str(rho))
            bw_str = f'{bw:.0e}' if bw < 0.01 else str(bw)
            config_label = f'L{config_idx+1}: ema={ema}, α={alpha}, bw={bw_str}, ρ={rho_str}'
            
            # Plot MSE
            ax_mse.scatter(l0_values, mse_values,
                          color=color, marker='D',
                          s=120, alpha=0.85, edgecolors='black', linewidth=0.5,
                          label=config_label)
            
            # Connect points with line
            sort_idx = np.argsort(l0_values)
            ax_mse.plot(l0_values[sort_idx], mse_values[sort_idx],
                       color=color, linewidth=1.5, alpha=0.5)
            
            # Plot EV
            ax_ev.scatter(l0_values, ev_values,
                         color=color, marker='D',
                         s=120, alpha=0.85, edgecolors='black', linewidth=0.5,
                         label=config_label)
            
            ax_ev.plot(l0_values[sort_idx], ev_values[sort_idx],
                      color=color, linewidth=1.5, alpha=0.5)
        
        # Find and highlight global Pareto frontier for all methods combined
        # Collect all points for MSE Pareto
        all_mse_l0 = []
        all_mse_vals = []
        all_mse_types = []  # 'baseline' or 'lagrangian'
        
        for sae_type in ['topk', 'batch_topk']:
            for run_data in baseline_data.get(sae_type, []):
                if layer_name in run_data['layers']:
                    l0 = run_data['layers'][layer_name]['l0']
                    mse = run_data['layers'][layer_name]['mse']
                    if min_l0 <= l0 <= max_l0:
                        all_mse_l0.append(l0)
                        all_mse_vals.append(mse)
                        all_mse_types.append('baseline')
        
        for l0, mse in all_lagrangian_mse_points:
            all_mse_l0.append(l0)
            all_mse_vals.append(mse)
            all_mse_types.append('lagrangian')
        
        if all_mse_l0:
            all_mse_l0 = np.array(all_mse_l0)
            all_mse_vals = np.array(all_mse_vals)
            is_pareto = find_pareto_frontier(all_mse_l0, all_mse_vals, minimize_x=True, minimize_y=True)
            
            if np.any(is_pareto):
                pareto_l0 = all_mse_l0[is_pareto]
                pareto_mse = all_mse_vals[is_pareto]
                sort_idx = np.argsort(pareto_l0)
                ax_mse.plot(pareto_l0[sort_idx], pareto_mse[sort_idx],
                           color='black', linewidth=3, alpha=0.8, linestyle='--',
                           label='Global Pareto', zorder=10)
        
        # Collect all points for EV Pareto
        all_ev_l0 = []
        all_ev_vals = []
        
        for sae_type in ['topk', 'batch_topk']:
            for run_data in baseline_data.get(sae_type, []):
                if layer_name in run_data['layers']:
                    l0 = run_data['layers'][layer_name]['l0']
                    ev = run_data['layers'][layer_name]['explained_variance']
                    if min_l0 <= l0 <= max_l0:
                        all_ev_l0.append(l0)
                        all_ev_vals.append(ev)
        
        for l0, ev in all_lagrangian_ev_points:
            all_ev_l0.append(l0)
            all_ev_vals.append(ev)
        
        if all_ev_l0:
            all_ev_l0 = np.array(all_ev_l0)
            all_ev_vals = np.array(all_ev_vals)
            is_pareto = find_pareto_frontier(all_ev_l0, -all_ev_vals, minimize_x=True, minimize_y=True)
            
            if np.any(is_pareto):
                pareto_l0 = all_ev_l0[is_pareto]
                pareto_ev = all_ev_vals[is_pareto]
                sort_idx = np.argsort(pareto_l0)
                ax_ev.plot(pareto_l0[sort_idx], pareto_ev[sort_idx],
                          color='black', linewidth=3, alpha=0.8, linestyle='--',
                          label='Global Pareto', zorder=10)
        
        # Configure MSE plot
        ax_mse.set_xlabel('L0 Sparsity', fontsize=14)
        ax_mse.set_ylabel('MSE (lower is better)', fontsize=14)
        ax_mse.set_title(f'MSE vs L0 - {layer_name}', fontsize=16)
        ax_mse.grid(True, alpha=0.3)
        ax_mse.tick_params(axis='both', labelsize=12)
        ax_mse.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Configure EV plot
        ax_ev.set_xlabel('L0 Sparsity', fontsize=14)
        ax_ev.set_ylabel('Explained Variance (higher is better)', fontsize=14)
        ax_ev.set_title(f'Explained Variance vs L0 - {layer_name}', fontsize=16)
        ax_ev.grid(True, alpha=0.3)
        ax_ev.tick_params(axis='both', labelsize=12)
        ax_ev.legend(loc='lower right', fontsize=8, framealpha=0.9)
        
        plt.suptitle(f'Lagrangian SAE (Top {top_n_configs}) vs Baselines - {layer_name}', 
                    fontsize=18, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        layer_display_name = layer_name.replace('.', '_')
        output_path = output_dir / f"lagrangian_vs_baselines_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"  Saved comparison plot to: {output_path}")
        plt.close()


def perform_comprehensive_analysis(
    data_by_config: Dict[Tuple, List[Dict]],
    layers: List[str],
    max_l0: float = 140.0,
    min_l0: float = 10.0
) -> Dict[str, Any]:
    """
    Perform comprehensive hyperparameter analysis to find optimal configurations.
    
    Includes:
    - Per-hyperparameter impact analysis
    - Best configuration per (layer, target_l0)
    - Global Pareto analysis
    - Hyperparameter preference counting
    - Recommendations
    
    Args:
        data_by_config: Dictionary mapping config tuples to run data
        layers: List of layer names
        max_l0: Maximum L0 threshold for filtering
        min_l0: Minimum L0 threshold for filtering
    
    Returns:
        Dictionary containing all analysis results
    """
    from collections import defaultdict
    
    print("\n" + "=" * 120)
    print("COMPREHENSIVE HYPERPARAMETER ANALYSIS")
    print("=" * 120)
    
    # =========================================================================
    # Collect all results into a flat structure
    # =========================================================================
    all_results = []
    
    for config_key, runs in data_by_config.items():
        for run_data in runs:
            for layer_name in run_data['layers']:
                l0 = run_data['layers'][layer_name]['l0']
                mse = run_data['layers'][layer_name]['mse']
                ev = run_data['layers'][layer_name]['explained_variance']
                k = run_data['target_l0']
                
                if min_l0 <= l0 <= max_l0:
                    l0_error = abs(l0 - k) / k  # Relative L0 error
                    all_results.append({
                        'config_key': config_key,
                        'l0_ema_momentum': config_key[0],
                        'alpha_max': config_key[1],
                        'bandwidth': config_key[2],
                        'rho_quadratic': config_key[3],
                        'layer': layer_name,
                        'target_l0': k,
                        'l0': l0,
                        'mse': mse,
                        'ev': ev,
                        'l0_error': l0_error,
                    })
    
    if not all_results:
        print("No data available for analysis.")
        return {}
    
    # =========================================================================
    # ANALYSIS A: Per-Hyperparameter Impact
    # =========================================================================
    print("\n" + "-" * 120)
    print("ANALYSIS A: PER-HYPERPARAMETER IMPACT")
    print("-" * 120)
    print("\nFor each hyperparameter value, showing average metrics across all layers and K values:\n")
    
    hyperparameter_impact = {}
    
    for param_name, param_values in HYPERPARAMETER_GRID.items():
        print(f"\n{param_name}:")
        print("-" * 80)
        
        param_stats = {}
        for value in param_values:
            # Filter results for this parameter value
            filtered = [r for r in all_results if r[param_name] == value]
            
            if filtered:
                mse_vals = [r['mse'] for r in filtered]
                ev_vals = [r['ev'] for r in filtered]
                l0_err_vals = [r['l0_error'] for r in filtered]
                
                stats = {
                    'count': len(filtered),
                    'mse_mean': np.mean(mse_vals),
                    'mse_std': np.std(mse_vals),
                    'ev_mean': np.mean(ev_vals),
                    'ev_std': np.std(ev_vals),
                    'l0_error_mean': np.mean(l0_err_vals),
                    'l0_error_std': np.std(l0_err_vals),
                }
                param_stats[value] = stats
                
                value_str = '0' if value == 0 else (f'{value:.0e}' if value < 0.01 else str(value))
                print(f"  {value_str:>8}: MSE={stats['mse_mean']:.6f}±{stats['mse_std']:.6f}, "
                      f"EV={stats['ev_mean']:.4f}±{stats['ev_std']:.4f}, "
                      f"L0_err={stats['l0_error_mean']*100:.1f}%±{stats['l0_error_std']*100:.1f}% "
                      f"(n={stats['count']})")
        
        # Find best value for each metric
        if param_stats:
            best_mse_val = min(param_stats.keys(), key=lambda v: param_stats[v]['mse_mean'])
            best_ev_val = max(param_stats.keys(), key=lambda v: param_stats[v]['ev_mean'])
            best_l0_val = min(param_stats.keys(), key=lambda v: param_stats[v]['l0_error_mean'])
            
            def fmt(v):
                return '0' if v == 0 else (f'{v:.0e}' if v < 0.01 else str(v))
            
            print(f"\n  BEST: MSE→{fmt(best_mse_val)}, EV→{fmt(best_ev_val)}, L0_precision→{fmt(best_l0_val)}")
            
            hyperparameter_impact[param_name] = {
                'stats': param_stats,
                'best_mse': best_mse_val,
                'best_ev': best_ev_val,
                'best_l0': best_l0_val,
            }
    
    # =========================================================================
    # ANALYSIS B: Best Configuration per (Layer, Target_L0)
    # =========================================================================
    print("\n" + "-" * 120)
    print("ANALYSIS B: BEST CONFIGURATION PER (LAYER, TARGET_L0)")
    print("-" * 120)
    
    best_configs = {}
    config_wins = defaultdict(lambda: {'mse': 0, 'ev': 0, 'l0': 0, 'combined': 0})
    
    for layer_name in layers:
        best_configs[layer_name] = {}
        layer_results = [r for r in all_results if r['layer'] == layer_name]
        
        if not layer_results:
            continue
        
        print(f"\n{layer_name}:")
        
        for k in VALID_K_VALUES:
            k_results = [r for r in layer_results if r['target_l0'] == k]
            if not k_results:
                continue
            
            # Best by MSE
            best_mse = min(k_results, key=lambda x: x['mse'])
            # Best by EV
            best_ev = max(k_results, key=lambda x: x['ev'])
            # Best by L0 precision
            best_l0 = min(k_results, key=lambda x: x['l0_error'])
            
            # Combined score: normalize MSE and L0 error, then sum
            # Lower is better for both
            mse_vals = np.array([r['mse'] for r in k_results])
            l0_err_vals = np.array([r['l0_error'] for r in k_results])
            
            # Min-max normalization
            mse_norm = (mse_vals - mse_vals.min()) / (mse_vals.max() - mse_vals.min() + 1e-10)
            l0_norm = (l0_err_vals - l0_err_vals.min()) / (l0_err_vals.max() - l0_err_vals.min() + 1e-10)
            combined_scores = mse_norm + l0_norm
            
            best_combined_idx = np.argmin(combined_scores)
            best_combined = k_results[best_combined_idx]
            
            best_configs[layer_name][k] = {
                'best_mse': best_mse,
                'best_ev': best_ev,
                'best_l0': best_l0,
                'best_combined': best_combined,
            }
            
            # Count wins
            config_wins[best_mse['config_key']]['mse'] += 1
            config_wins[best_ev['config_key']]['ev'] += 1
            config_wins[best_l0['config_key']]['l0'] += 1
            config_wins[best_combined['config_key']]['combined'] += 1
            
            print(f"  K={k:3d}: MSE→{get_config_display_name(best_mse['config_key'])} "
                  f"(MSE={best_mse['mse']:.6f})")
            print(f"         EV→{get_config_display_name(best_ev['config_key'])} "
                  f"(EV={best_ev['ev']:.4f})")
            print(f"         L0→{get_config_display_name(best_l0['config_key'])} "
                  f"(L0_err={best_l0['l0_error']*100:.1f}%)")
            print(f"         Combined→{get_config_display_name(best_combined['config_key'])}")
    
    # =========================================================================
    # ANALYSIS C: Global Pareto Analysis (MSE vs L0 error trade-off)
    # =========================================================================
    print("\n" + "-" * 120)
    print("ANALYSIS C: GLOBAL PARETO ANALYSIS (MSE vs L0 ERROR TRADE-OFF)")
    print("-" * 120)
    
    pareto_configs = {}
    pareto_appearance_count = defaultdict(int)
    
    for layer_name in layers:
        layer_results = [r for r in all_results if r['layer'] == layer_name]
        if not layer_results:
            continue
        
        pareto_configs[layer_name] = {}
        print(f"\n{layer_name}:")
        
        for k in VALID_K_VALUES:
            k_results = [r for r in layer_results if r['target_l0'] == k]
            if not k_results:
                continue
            
            # Find Pareto-optimal configurations for MSE vs L0_error
            mse_vals = np.array([r['mse'] for r in k_results])
            l0_err_vals = np.array([r['l0_error'] for r in k_results])
            
            is_pareto = find_pareto_frontier(mse_vals, l0_err_vals, 
                                            minimize_x=True, minimize_y=True)
            
            pareto_points = [k_results[i] for i in range(len(k_results)) if is_pareto[i]]
            pareto_configs[layer_name][k] = pareto_points
            
            print(f"  K={k}: {len(pareto_points)} Pareto-optimal configs:")
            for p in sorted(pareto_points, key=lambda x: x['mse']):
                pareto_appearance_count[p['config_key']] += 1
                print(f"    - {get_config_display_name(p['config_key'])}: "
                      f"MSE={p['mse']:.6f}, L0_err={p['l0_error']*100:.1f}%")
    
    # =========================================================================
    # ANALYSIS D: Hyperparameter Preference Counting
    # =========================================================================
    print("\n" + "-" * 120)
    print("ANALYSIS D: HYPERPARAMETER PREFERENCE COUNTING")
    print("-" * 120)
    print("\nCounting how often each configuration wins (best by any metric):\n")
    
    # Sort configs by total wins
    config_total_wins = {
        k: v['mse'] + v['ev'] + v['l0'] + v['combined']
        for k, v in config_wins.items()
    }
    sorted_configs = sorted(config_total_wins.items(), key=lambda x: -x[1])[:15]
    
    print("Top 15 Configurations by Total Wins:")
    print(f"{'Configuration':<65} {'MSE':>5} {'EV':>5} {'L0':>5} {'Comb':>5} {'Total':>6} {'Pareto':>6}")
    print("-" * 105)
    
    for config_key, total in sorted_configs:
        wins = config_wins[config_key]
        pareto_count = pareto_appearance_count.get(config_key, 0)
        print(f"{get_config_display_name(config_key):<65} "
              f"{wins['mse']:>5} {wins['ev']:>5} {wins['l0']:>5} {wins['combined']:>5} "
              f"{total:>6} {pareto_count:>6}")
    
    # Count wins per hyperparameter value
    print("\n\nWins by Hyperparameter Value (sum of wins for all configs with this value):")
    print("-" * 80)
    
    hyperparam_wins = {}
    for param_name, param_values in HYPERPARAMETER_GRID.items():
        wins_by_value = {}
        for value in param_values:
            total_wins = sum(
                config_total_wins.get(ck, 0) 
                for ck in config_total_wins 
                if ck[list(HYPERPARAMETER_GRID.keys()).index(param_name)] == value
            )
            wins_by_value[value] = total_wins
        
        best_value = max(wins_by_value.items(), key=lambda x: x[1])
        hyperparam_wins[param_name] = {'wins_by_value': wins_by_value, 'best': best_value[0]}
        
        print(f"\n{param_name}:")
        for value, wins in sorted(wins_by_value.items(), key=lambda x: -x[1]):
            value_str = '0' if value == 0 else (f'{value:.0e}' if value < 0.01 else str(value))
            marker = " ← BEST" if value == best_value[0] else ""
            print(f"  {value_str:>8}: {wins:4d} wins{marker}")
    
    # =========================================================================
    # ANALYSIS E: Recommendations
    # =========================================================================
    print("\n" + "-" * 120)
    print("ANALYSIS E: RECOMMENDATIONS")
    print("-" * 120)
    
    # Recommend based on different criteria
    print("\n1. RECOMMENDED CONFIGURATION PER TARGET L0 (by combined score wins):")
    print("-" * 80)
    
    for k in VALID_K_VALUES:
        # Find config with most combined wins for this K
        k_combined_wins = defaultdict(int)
        for layer_name in layers:
            if layer_name in best_configs and k in best_configs[layer_name]:
                best = best_configs[layer_name][k]['best_combined']
                k_combined_wins[best['config_key']] += 1
        
        if k_combined_wins:
            best_config = max(k_combined_wins.items(), key=lambda x: x[1])
            print(f"  K={k:3d}: {get_config_display_name(best_config[0])} "
                  f"(won in {best_config[1]}/{len(layers)} layers)")
    
    print("\n2. OVERALL BEST CONFIGURATIONS:")
    print("-" * 80)
    
    # Best average MSE
    avg_mse_by_config = defaultdict(list)
    avg_ev_by_config = defaultdict(list)
    avg_l0err_by_config = defaultdict(list)
    
    for r in all_results:
        avg_mse_by_config[r['config_key']].append(r['mse'])
        avg_ev_by_config[r['config_key']].append(r['ev'])
        avg_l0err_by_config[r['config_key']].append(r['l0_error'])
    
    best_avg_mse = min(avg_mse_by_config.items(), key=lambda x: np.mean(x[1]))
    best_avg_ev = max(avg_ev_by_config.items(), key=lambda x: np.mean(x[1]))
    best_avg_l0 = min(avg_l0err_by_config.items(), key=lambda x: np.mean(x[1]))
    
    print(f"  Best Average MSE:          {get_config_display_name(best_avg_mse[0])}")
    print(f"                             (avg MSE = {np.mean(best_avg_mse[1]):.6f})")
    print(f"  Best Average EV:           {get_config_display_name(best_avg_ev[0])}")
    print(f"                             (avg EV = {np.mean(best_avg_ev[1]):.4f})")
    print(f"  Best Average L0 Precision: {get_config_display_name(best_avg_l0[0])}")
    print(f"                             (avg L0_err = {np.mean(best_avg_l0[1])*100:.1f}%)")
    
    print("\n3. RECOMMENDED HYPERPARAMETER VALUES (based on win counting):")
    print("-" * 80)
    
    def fmt(v):
        return '0' if v == 0 else (f'{v:.0e}' if v < 0.01 else str(v))
    
    for param_name in HYPERPARAMETER_GRID.keys():
        best_val = hyperparam_wins[param_name]['best']
        print(f"  {param_name}: {fmt(best_val)}")
    
    # Construct final recommended config
    recommended_config = (
        hyperparam_wins['l0_ema_momentum']['best'],
        hyperparam_wins['alpha_max']['best'],
        hyperparam_wins['bandwidth']['best'],
        hyperparam_wins['rho_quadratic']['best'],
    )
    
    print(f"\n  FINAL RECOMMENDED CONFIG: {get_config_display_name(recommended_config)}")
    
    print("\n4. MOST PARETO-OPTIMAL CONFIGURATIONS (appears on Pareto frontier most often):")
    print("-" * 80)
    
    sorted_pareto = sorted(pareto_appearance_count.items(), key=lambda x: -x[1])[:10]
    for config_key, count in sorted_pareto:
        print(f"  {get_config_display_name(config_key)}: {count} Pareto frontier appearances")
    
    print("\n" + "=" * 120)
    
    return {
        'hyperparameter_impact': hyperparameter_impact,
        'best_configs': best_configs,
        'config_wins': dict(config_wins),
        'pareto_configs': pareto_configs,
        'pareto_appearance_count': dict(pareto_appearance_count),
        'hyperparam_wins': hyperparam_wins,
        'recommended_config': recommended_config,
        'best_avg_mse': best_avg_mse[0],
        'best_avg_ev': best_avg_ev[0],
        'best_avg_l0': best_avg_l0[0],
    }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot Pareto curves for Lagrangian SAE hyperparameter sweeps"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Wandb project for sweep runs (default: {wandb_entity}/gpt2-small-sweep)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/sweep",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--max-l0",
        type=float,
        default=140.0,
        help="Maximum L0 threshold for filtering (default: 140)"
    )
    parser.add_argument(
        "--min-l0",
        type=float,
        default=10.0,
        help="Minimum L0 threshold for filtering (default: 10)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip creating summary heatmaps"
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip creating overlay plots"
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip comprehensive hyperparameter analysis"
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only run comprehensive analysis (skip all plots)"
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Compare top Lagrangian configs against TopK and BatchTopK baselines"
    )
    parser.add_argument(
        "--baseline-project",
        type=str,
        default=None,
        help="Wandb project for baseline runs (default: {wandb_entity}/gpt2-small)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of top Lagrangian configs to compare against baselines (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Collect data
    print("=" * 100)
    print("Collecting Lagrangian SAE sweep data...")
    print("=" * 100)
    data_by_config, layers = collect_sweep_metrics_data(args.project)
    
    if not data_by_config:
        print("\nNo sweep data found. Make sure runs exist in the specified project.")
        print("Expected run name format: lagrangian_target_l0_{K}_l0_ema_momentum_{ema}_...")
        return
    
    # Skip plots if analysis-only
    if not args.analysis_only:
        # Create Pareto plots
        print("\n" + "=" * 100)
        print("Creating Pareto curve plots...")
        print("=" * 100)
        plot_sweep_pareto_curves(
            data_by_config, layers, Path(args.output_dir),
            max_l0=args.max_l0, min_l0=args.min_l0
        )
        
        # Create summary heatmaps
        if not args.no_summary:
            print("\n" + "=" * 100)
            print("Creating summary heatmaps...")
            print("=" * 100)
            plot_sweep_summary(
                data_by_config, layers, Path(args.output_dir),
                max_l0=args.max_l0, min_l0=args.min_l0
            )
        
        # Create overlay plots
        if not args.no_overlay:
            print("\n" + "=" * 100)
            print("Creating overlay Pareto plots...")
            print("=" * 100)
            plot_overlay_pareto_curves(
                data_by_config, layers, Path(args.output_dir),
                max_l0=args.max_l0, min_l0=args.min_l0
            )
        
        # Print basic summary
        print_sweep_summary(data_by_config, layers, max_l0=args.max_l0, min_l0=args.min_l0)
    
    # Perform comprehensive analysis
    if not args.no_analysis:
        analysis_results = perform_comprehensive_analysis(
            data_by_config, layers,
            max_l0=args.max_l0, min_l0=args.min_l0
        )
    
    # Compare against baselines if requested
    if args.compare_baselines:
        print("\n" + "=" * 100)
        print("Comparing against TopK and BatchTopK baselines...")
        print("=" * 100)
        
        # Collect baseline data
        baseline_data, baseline_layers = collect_baseline_metrics_data(args.baseline_project)
        
        if not baseline_data.get('topk') and not baseline_data.get('batch_topk'):
            print("\nNo baseline data found. Make sure runs exist in the baseline project.")
            print("Expected run name formats: topk_k_{K}, batch_topk_k_{K}")
        else:
            # Create comparison plots
            print("\n" + "=" * 100)
            print("Creating baseline comparison plots...")
            print("=" * 100)
            plot_comparison_pareto_curves(
                data_by_config, baseline_data, layers,
                top_n_configs=args.top_n,
                output_dir=Path(args.output_dir),
                max_l0=args.max_l0, min_l0=args.min_l0
            )
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()


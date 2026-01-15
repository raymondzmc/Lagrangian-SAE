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
    # Check target_l0
    target_pattern = f"target_l0_{target_l0}"
    if target_pattern not in run_name:
        return False
    
    # Check each hyperparameter
    for param_name, value in config.items():
        formatted_value = format_param_value(param_name, value)
        param_pattern = f"{param_name}_{formatted_value}"
        
        # Handle scientific notation matching (0.001 might be 1e-03 or 0.001)
        if param_pattern not in run_name:
            # Try alternate formats
            if value == 0:
                alt_patterns = [f"{param_name}_0", f"{param_name}_0.0"]
            elif value < 0.01:
                alt_patterns = [
                    f"{param_name}_{value:.0e}",
                    f"{param_name}_{value}",
                    f"{param_name}_{value:.3f}",
                ]
            else:
                alt_patterns = [f"{param_name}_{value}", f"{param_name}_{value:.1f}"]
            
            if not any(p in run_name for p in alt_patterns):
                return False
    
    return True


def collect_sweep_metrics_data(
    project: str = "raymondl/gpt2-small-sweep"
) -> Tuple[Dict[Tuple, List[Dict]], List[str]]:
    """
    Collect metrics data for all Lagrangian SAE sweep configurations.
    
    Args:
        project: Wandb project name for sweep runs
    
    Returns:
        Tuple of (data_by_config, layers) where data_by_config is a dictionary
        mapping configuration tuples to lists of run data
    """
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


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot Pareto curves for Lagrangian SAE hyperparameter sweeps"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="raymondl/gpt2-small-sweep",
        help="Wandb project for sweep runs"
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
    
    # Print summary
    print_sweep_summary(data_by_config, layers, max_l0=args.max_l0, min_l0=args.min_l0)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()


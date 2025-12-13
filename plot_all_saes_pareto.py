#!/usr/bin/env python3
"""
Plot Pareto curves for all SAE types: TopK, BatchTopK, and Lagrangian.
"""

import re
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

from settings import settings
from utils.io import load_metrics_from_wandb


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


def collect_all_metrics_data(projects: List[str] = None) -> Dict[str, List[Dict]]:
    """
    Collect metrics data for all SAE types from multiple projects.
    
    Args:
        projects: List of project names to collect data from. If None, uses default projects.
    
    Returns:
        Dictionary with SAE type keys, each containing list of run data
    """
    if projects is None:
        projects = [
            "raymondl/gpt2-small"  # GPT-2 project with relu and gated runs
        ]
    
    print(f"Collecting metrics data from {len(projects)} Wandb projects...")
    for project in projects:
        print(f"  - {project}")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs from all projects
    all_runs = []
    for project in projects:
        print(f"\nFetching runs from {project}...")
        project_runs = list(api.runs(project))
        print(f"  Found {len(project_runs)} runs")
        # Add project info to each run for tracking
        for run in project_runs:
            run._project_name = project
        all_runs.extend(project_runs)
    
    print(f"\nTotal runs across all projects: {len(all_runs)}")
    runs = all_runs
    
    # Collect data by SAE type - TopK, BatchTopK, and Lagrangian for GPT-2
    data = {
        'topk': [],         # runs with "topk_k_{K}"
        'batch_topk': [],   # runs with "batch_topk_k_{K}"
        'lagrangian': [],   # runs with "lagrangian_target_l0_{K}"
    }
    
    # Valid K values
    valid_k_values = {16, 32, 64, 128}
    
    # Track layer names
    all_layers = set()
    
    # First pass: collect runs by name to handle duplicates (keep latest)
    runs_by_name = {}
    for run in runs:
        name = run.name
        # Check if this run matches our patterns
        topk_match = re.match(r'^topk_k_(\d+)$', name)
        batch_topk_match = re.match(r'^batch_topk_k_(\d+)$', name)
        lagrangian_match = re.match(r'^lagrangian_target_l0_(\d+)$', name)
        
        if topk_match or batch_topk_match or lagrangian_match:
            if name not in runs_by_name:
                runs_by_name[name] = run
            else:
                # Keep the latest run (compare created_at timestamps)
                existing_run = runs_by_name[name]
                if run.created_at > existing_run.created_at:
                    runs_by_name[name] = run
    
    print(f"\nAfter deduplication: {len(runs_by_name)} unique runs")
    
    for run in runs_by_name.values():
        name = run.name
        
        # Determine SAE type based on run name patterns
        sae_type = None
        k_value = None
        
        topk_match = re.match(r'^topk_k_(\d+)$', name)
        batch_topk_match = re.match(r'^batch_topk_k_(\d+)$', name)
        lagrangian_match = re.match(r'^lagrangian_target_l0_(\d+)$', name)
        
        if topk_match:
            k_value = int(topk_match.group(1))
            if k_value in valid_k_values:
                sae_type = 'topk'
        elif batch_topk_match:
            k_value = int(batch_topk_match.group(1))
            if k_value in valid_k_values:
                sae_type = 'batch_topk'
        elif lagrangian_match:
            k_value = int(lagrangian_match.group(1))
            if k_value in valid_k_values:
                sae_type = 'lagrangian'
        
        if sae_type is None:
            continue
            
        # Load metrics for this run
        run_project = getattr(run, '_project_name', projects[0])  # Fallback to first project
        print(f"  Loading metrics for {run.name} ({run.id}) from {run_project} - Type: {sae_type}")
        metrics = load_metrics_from_wandb(run.id, run_project)
        
        if metrics:
            # Store per-layer metrics
            run_data = {
                'run_name': run.name,
                'run_id': run.id,
                'param_value': k_value,  # K value extracted during pattern matching
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
            
            data[sae_type].append(run_data)
    
    # Sort by average L0 sparsity for consistency
    for sae_type in data:
        if data[sae_type]:
            data[sae_type] = sorted(data[sae_type], 
                                    key=lambda x: np.mean([m['l0'] for m in x['layers'].values()]))
    
    print(f"\nCollected data summary:")
    for sae_type, runs in data.items():
        print(f"  {sae_type}: {len(runs)} runs")
    print(f"Found {len(all_layers)} layers: {sorted(all_layers)}")
    
    return data, sorted(all_layers)


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
                    # Both objectives should be minimized
                    if x_values[j] <= x_values[i] and y_values[j] <= y_values[i]:
                        if x_values[j] < x_values[i] or y_values[j] < y_values[i]:
                            is_pareto[i] = False
                            break
                elif minimize_x and not minimize_y:
                    # Minimize x, maximize y
                    if x_values[j] <= x_values[i] and y_values[j] >= y_values[i]:
                        if x_values[j] < x_values[i] or y_values[j] > y_values[i]:
                            is_pareto[i] = False
                            break
    
    return is_pareto


def plot_all_pareto_curves(data: Dict[str, List[Dict]], layers: List[str], 
                           output_dir: Path = Path("plots"),
                           max_mse: float = float('inf'), max_l0: float = 34.0,
                           min_mse: float = 0.0, min_l0: float = 4.0,
                           use_log_scale: bool = False):
    """
    Create Pareto curve plots for all SAE types.
    
    Args:
        data: Dictionary with SAE type data
        layers: List of layer names
        output_dir: Output directory for plots
        max_mse: Maximum MSE threshold (default: inf - no filtering)
        max_l0: Maximum L0 threshold (default: inf - no filtering)
        use_log_scale: Whether to use log scale for axes (default: False)
    """
    output_dir.mkdir(exist_ok=True)
    
    if max_mse == float('inf') and max_l0 == 34.0 and min_mse == 0.0 and min_l0 == 4.0:
        print(f"\nUsing default L0 filtering: L0 in [{min_l0}, {max_l0}]")
    elif max_mse == float('inf') and max_l0 == float('inf') and min_mse == 0.0 and min_l0 == 0.0:
        print(f"\nPlotting all data points (no filtering)")
    else:
        print(f"\nFiltering: MSE in [{min_mse}, {max_mse}], L0 in [{min_l0}, {max_l0}]")
    
    # Color scheme for different SAE types - more distinct colors
    colors = {
        'topk': '#1f77b4',                            # Blue
        'batch_topk': '#2ca02c',                      # Green
        'lagrangian': '#d62728',                      # Red
    }
    
    # Marker styles - more distinct shapes
    markers = {
        'topk': 'o',                                  # Circle
        'batch_topk': 's',                            # Square
        'lagrangian': 'D',                            # Diamond
    }
    
    # Labels for legend
    labels = {
        'topk': 'TopK',
        'batch_topk': 'BatchTopK',
        'lagrangian': 'Lagrangian (ours)',
    }
    
    # Track filtered statistics
    total_filtered = 0
    filtered_by_type = {k: 0 for k in data.keys()}
    
    # Create a figure for each layer
    for layer_idx, layer_name in enumerate(layers):
        print(f"\nProcessing layer: {layer_name}")
        layer_filtered = 0
        
        # Create figure with 2 subplots for this layer - even larger size for better visibility
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        # Plot 1: MSE vs L0 (minimize both)
        ax1 = axes[0]
        for sae_type in ['topk', 'batch_topk', 'lagrangian']:
            if data.get(sae_type):
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                mse_values = []
                run_names = []
                param_labels = []  # Will store sparsity_coeff for HC or k for Top-K
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        
                        # Apply filters
                        if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                            l0_values.append(l0)
                            mse_values.append(mse)
                            run_names.append(run_data['run_name'])
                            
                            # Get parameter for labeling
                            param_labels.append(run_data.get('param_value', None))
                        else:
                            layer_filtered += 1
                            filtered_by_type[sae_type] += 1
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                mse_values = np.array(mse_values)
                
                # Find Pareto frontier (minimize both MSE and L0)
                is_pareto = find_pareto_frontier(l0_values, mse_values, 
                                                minimize_x=True, minimize_y=True)
                
                # Plot all points
                ax1.scatter(l0_values, mse_values, 
                           color=colors[sae_type], marker=markers[sae_type],
                           alpha=0.7, s=150, label=f'{labels[sae_type]}')
                
                # Add parameter labels for all points
                for i, (x, y, param) in enumerate(zip(l0_values, mse_values, param_labels)):
                    if param is not None:
                        # Format label for display based on parameter type
                        if isinstance(param, int):
                            label = f'{param}'
                        elif param >= 0.01:
                            label = f'{param:.2f}'
                        elif param >= 0.001:
                            label = f'{param:.3f}'
                        else:
                            label = f'{param:.0e}'
                        ax1.annotate(label, (x, y), 
                                   xytext=(3, 3), textcoords='offset points',
                                   fontsize=14, alpha=0.6, color=colors[sae_type])
                
                # Highlight and connect Pareto frontier points
                if np.any(is_pareto):
                    pareto_l0 = l0_values[is_pareto]
                    pareto_mse = mse_values[is_pareto]
                    
                    # Sort for line plotting
                    sort_idx = np.argsort(pareto_l0)
                    pareto_l0 = pareto_l0[sort_idx]
                    pareto_mse = pareto_mse[sort_idx]
                    
                    ax1.plot(pareto_l0, pareto_mse, 
                            color=colors[sae_type], linewidth=2, alpha=0.8)
                    ax1.scatter(pareto_l0, pareto_mse,
                               color=colors[sae_type], marker=markers[sae_type],
                               s=180, edgecolors='black', linewidth=2, zorder=5)
        
        ax1.set_xlabel('L0 Sparsity', fontsize=28, labelpad=15)
        ax1.set_ylabel('MSE ← (better)', fontsize=28, labelpad=15)
        ax1.set_title('MSE vs L0', fontsize=32, pad=20)
        if use_log_scale:
            ax1.set_xscale('log')
            ax1.set_yscale('log')
        ax1.legend(loc='upper right', fontsize=24)
        ax1.tick_params(axis='both', labelsize=26)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Explained Variance vs L0 (minimize L0, maximize explained variance)
        ax2 = axes[1]
        for sae_type in ['topk', 'batch_topk', 'lagrangian']:
            if data.get(sae_type):
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                ev_values = []
                param_labels = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        ev = run_data['layers'][layer_name]['explained_variance']
                        
                        # Apply filters
                        if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                            l0_values.append(l0)
                            ev_values.append(ev)
                            
                            # Get parameter for labeling
                            param_labels.append(run_data.get('param_value', None))
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                ev_values = np.array(ev_values)
                
                # Find Pareto frontier (minimize L0, maximize explained variance)
                is_pareto = find_pareto_frontier(l0_values, -ev_values,  # Negate EV to find max
                                                minimize_x=True, minimize_y=True)
                
                # Plot all points
                ax2.scatter(l0_values, ev_values,
                           color=colors[sae_type], marker=markers[sae_type],
                           alpha=0.7, s=150, label=f'{labels[sae_type]}')
                
                # Add parameter labels for all points
                for i, (x, y, param) in enumerate(zip(l0_values, ev_values, param_labels)):
                    if param is not None:
                        # Format label for display based on parameter type
                        if isinstance(param, int):
                            label = f'{param}'
                        elif param >= 0.01:
                            label = f'{param:.2f}'
                        elif param >= 0.001:
                            label = f'{param:.3f}'
                        else:
                            label = f'{param:.0e}'
                        ax2.annotate(label, (x, y), 
                                   xytext=(3, 3), textcoords='offset points',
                                   fontsize=14, alpha=0.6, color=colors[sae_type])
                
                # Highlight and connect Pareto frontier points
                if np.any(is_pareto):
                    pareto_l0 = l0_values[is_pareto]
                    pareto_ev = ev_values[is_pareto]
                    
                    # Sort for line plotting
                    sort_idx = np.argsort(pareto_l0)
                    pareto_l0 = pareto_l0[sort_idx]
                    pareto_ev = pareto_ev[sort_idx]
                    
                    ax2.plot(pareto_l0, pareto_ev,
                            color=colors[sae_type], linewidth=2, alpha=0.8)
                    ax2.scatter(pareto_l0, pareto_ev,
                               color=colors[sae_type], marker=markers[sae_type],
                               s=180, edgecolors='black', linewidth=2, zorder=5)
        
        ax2.set_xlabel('L0 Sparsity', fontsize=28, labelpad=15)
        ax2.set_ylabel('Explained Variance → (better)', fontsize=28, labelpad=15)
        ax2.set_title('Explained Variance vs L0', fontsize=32, pad=20)
        
        # Move y-axis label and ticks to the right side
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        
        if use_log_scale:
            ax2.set_xscale('log')
            # Don't use log scale for explained variance as it goes from 0 to 1
        ax2.legend(loc='lower right', fontsize=24)
        ax2.tick_params(axis='both', labelsize=26)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Adjust layout and save
        layer_display_name = layer_name.replace('.', '_')
        if max_mse == float('inf') and max_l0 == 34.0 and min_mse == 0.0 and min_l0 == 4.0:
            filter_str = f'Default L0 filtering: L0 ∈ [{min_l0}, {max_l0}]'
        elif max_mse == float('inf') and max_l0 == float('inf') and min_mse == 0.0 and min_l0 == 0.0:
            filter_str = 'No filtering'
        else:
            filter_str = f'MSE ∈ [{min_mse}, {max_mse}], L0 ∈ [{min_l0}, {max_l0}]'
        scale_str = ' (log scale)' if use_log_scale else ''
        # Removed suptitle as requested
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"all_saes_pareto_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved plot to: {output_path}")
        if layer_filtered > 0:
            print(f"  Filtered {layer_filtered} points from this layer")
        
        # Also save as SVG for vector graphics
        output_path_svg = output_dir / f"all_saes_pareto_{layer_display_name}.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()
        
        total_filtered += layer_filtered
    
    if total_filtered > 0:
        print(f"\nTotal points filtered: {total_filtered}")
        for sae_type, count in filtered_by_type.items():
            if count > 0:
                print(f"  {sae_type}: {count} points")


def plot_alive_dictionary_components(data: Dict[str, List[Dict]], layers: List[str], 
                                   output_dir: Path = Path("plots"),
                                   max_mse: float = float('inf'), max_l0: float = 34.0,
                                   min_mse: float = 0.0, min_l0: float = 4.0):
    """
    Create separate plots for Alive Dictionary Components vs L0 with all lines connected.
    
    Args:
        data: Dictionary with SAE type data
        layers: List of layer names
        output_dir: Output directory for plots
        max_mse: Maximum MSE threshold for filtering
        max_l0: Maximum L0 threshold for filtering
        min_mse: Minimum MSE threshold for filtering
        min_l0: Minimum L0 threshold for filtering
    """
    output_dir.mkdir(exist_ok=True)
    
    # Color scheme and markers (same as main plots)
    colors = {
        'topk': '#1f77b4',                            # Blue
        'batch_topk': '#2ca02c',                      # Green
        'lagrangian': '#d62728',                      # Red
    }
    
    markers = {
        'topk': 'o',                                  # Circle
        'batch_topk': 's',                            # Square
        'lagrangian': 'D',                            # Diamond
    }
    
    labels = {
        'topk': 'TopK',
        'batch_topk': 'BatchTopK',
        'lagrangian': 'Lagrangian (ours)',
    }
    
    # Create a figure for each layer
    for layer_idx, layer_name in enumerate(layers):
        print(f"\nProcessing alive components for layer: {layer_name}")
        
        # Create figure with single subplot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for sae_type in ['topk', 'batch_topk', 'lagrangian']:
            if data.get(sae_type):
                # Extract layer-specific data WITH FILTERING
                l0_values = []
                alive_values = []
                param_labels = []
                
                for run_data in data[sae_type]:
                    if layer_name in run_data['layers']:
                        l0 = run_data['layers'][layer_name]['l0']
                        mse = run_data['layers'][layer_name]['mse']
                        alive = run_data['layers'][layer_name]['alive_dict_components']
                        
                        # Apply filters
                        if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                            l0_values.append(l0)
                            alive_values.append(alive)
                            param_labels.append(run_data.get('param_value', None))
                
                if not l0_values:
                    continue
                    
                l0_values = np.array(l0_values)
                alive_values = np.array(alive_values)
                
                # Sort by L0 for line plotting
                sort_idx = np.argsort(l0_values)
                l0_sorted = l0_values[sort_idx]
                alive_sorted = alive_values[sort_idx]
                param_sorted = [param_labels[i] for i in sort_idx]
                
                # Plot all points
                ax.scatter(l0_sorted, alive_sorted,
                          color=colors[sae_type], marker=markers[sae_type],
                          alpha=0.7, s=150, label=f'{labels[sae_type]}', zorder=3)
                
                # Connect all points with lines
                ax.plot(l0_sorted, alive_sorted,
                       color=colors[sae_type], linewidth=3, alpha=0.8, zorder=2)
                
                # Add parameter labels for all points
                for i, (x, y, param) in enumerate(zip(l0_sorted, alive_sorted, param_sorted)):
                    if param is not None:
                        # Format label for display based on parameter type
                        if isinstance(param, int):
                            label = f'{param}'
                        elif param >= 0.01:
                            label = f'{param:.2f}'
                        elif param >= 0.001:
                            label = f'{param:.3f}'
                        else:
                            label = f'{param:.0e}'
                        ax.annotate(label, (x, y), 
                                   xytext=(3, 3), textcoords='offset points',
                                   fontsize=14, alpha=0.6, color=colors[sae_type])
        
        ax.set_xlabel('L0 Sparsity', fontsize=28, labelpad=15)
        ax.set_ylabel('Alive Dictionary Components → (better)', fontsize=28, labelpad=15)
        ax.set_title('Alive Dictionary Components vs L0', fontsize=32, pad=20)
        ax.legend(loc='lower right', fontsize=24)
        ax.tick_params(axis='both', labelsize=26)
        ax.grid(True, alpha=0.3, which='both')
        
        # Adjust layout and save
        layer_display_name = layer_name.replace('.', '_')
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"alive_components_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved alive components plot to: {output_path}")
        
        # Also save as SVG
        output_path_svg = output_dir / f"alive_components_{layer_display_name}.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()


def print_pareto_summary(data: Dict[str, List[Dict]], layers: List[str],
                        max_mse: float = float('inf'), max_l0: float = 34.0,
                        min_mse: float = 0.0, min_l0: float = 4.0):
    """Print a summary of the Pareto-optimal points for each layer and SAE type."""
    
    print("\n" + "=" * 80)
    if max_mse == float('inf') and max_l0 == 34.0 and min_mse == 0.0 and min_l0 == 4.0:
        filter_str = f'Default L0 filtering: L0 ∈ [{min_l0}, {max_l0}]'
    elif max_mse == float('inf') and max_l0 == float('inf') and min_mse == 0.0 and min_l0 == 0.0:
        filter_str = 'No filtering'
    else:
        filter_str = f'MSE ∈ [{min_mse}, {max_mse}], L0 ∈ [{min_l0}, {max_l0}]'
    print(f"PARETO FRONTIER SUMMARY - ALL SAE TYPES ({filter_str})")
    print("=" * 80)
    
    # Define display names
    display_names = {
        'topk': 'TopK',
        'batch_topk': 'BatchTopK',
        'lagrangian': 'Lagrangian',
    }
    
    for layer_name in layers:
        print(f"\n{'='*80}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*80}")
        
        for sae_type in ['topk', 'batch_topk', 'lagrangian']:
            if not data.get(sae_type):
                continue
            
            print(f"\n{display_names[sae_type]} SAE:")
            print("-" * 40)
            
            # Extract layer-specific data WITH FILTERING
            runs_data = []
            filtered_count = 0
            for run_data in data[sae_type]:
                if layer_name in run_data['layers']:
                    l0 = run_data['layers'][layer_name]['l0']
                    mse = run_data['layers'][layer_name]['mse']
                    
                    if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                        runs_data.append({
                            'name': run_data['run_name'],
                            'l0': l0,
                            'mse': mse,
                            'ev': run_data['layers'][layer_name]['explained_variance'],
                            'alive': run_data['layers'][layer_name]['alive_dict_components']
                        })
                    else:
                        filtered_count += 1
            
            if not runs_data:
                if max_mse != float('inf') or max_l0 != 34.0 or min_mse != 0.0 or min_l0 != 4.0:
                    print(f"  No data available after filtering ({filtered_count} runs filtered)")
                else:
                    print(f"  No data available")
                continue
            elif filtered_count > 0 and (max_mse != float('inf') or max_l0 != 34.0 or min_mse != 0.0 or min_l0 != 4.0):
                print(f"  ({filtered_count} runs filtered out)")
            
            # Calculate statistics
            l0_values = np.array([r['l0'] for r in runs_data])
            mse_values = np.array([r['mse'] for r in runs_data])
            ev_values = np.array([r['ev'] for r in runs_data])
            
            print(f"  Statistics ({len(runs_data)} runs):")
            print(f"    L0: {l0_values.mean():.2f} ± {l0_values.std():.2f}")
            print(f"    MSE: {mse_values.mean():.6f} ± {mse_values.std():.6f}")
            print(f"    EV: {ev_values.mean():.4f} ± {ev_values.std():.4f}")
            
            # Find Pareto points
            is_pareto_mse = find_pareto_frontier(l0_values, mse_values, 
                                                minimize_x=True, minimize_y=True)
            is_pareto_ev = find_pareto_frontier(l0_values, -ev_values,
                                               minimize_x=True, minimize_y=True)
            
            # Count Pareto optimal runs
            n_pareto = np.sum(is_pareto_mse | is_pareto_ev)
            if n_pareto > 0:
                print(f"  Pareto-optimal runs: {n_pareto}/{len(runs_data)}")
    
    print("\n" + "=" * 80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot Pareto curves for SAE types: topk, batch_topk, and lagrangian for GPT-2 experiments"
    )
    parser.add_argument(
        "--projects",
        type=str,
        nargs='+',
        default=None,
        help="Wandb projects to collect data from (default: raymondl/gpt2-small)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--max-mse",
        type=float,
        default=float('inf'),
        help="Maximum MSE threshold for filtering (default: inf - no filtering)"
    )
    parser.add_argument(
        "--max-l0",
        type=float,
        default=34,
        help="Maximum L0 threshold for filtering (default: 34)"
    )
    parser.add_argument(
        "--min-mse",
        type=float,
        default=0.0,
        help="Minimum MSE threshold for filtering (default: 0.0)"
    )
    parser.add_argument(
        "--min-l0",
        type=float,
        default=4.0,
        help="Minimum L0 threshold for filtering (default: 4.0)"
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for axes (default: False)"
    )
    
    args = parser.parse_args()
    
    # Handle log scale flag - simple now
    use_log_scale = args.log_scale
    
    # Collect data
    print("=" * 80)
    print("Collecting data for all SAE types...")
    print("=" * 80)
    data, layers = collect_all_metrics_data(args.projects)
    
    # Create plots with filtering
    print("\n" + "=" * 80)
    print("Creating Pareto curve plots...")
    print("=" * 80)
    plot_all_pareto_curves(data, layers, Path(args.output_dir), 
                          max_mse=args.max_mse, max_l0=args.max_l0,
                          min_mse=args.min_mse, min_l0=args.min_l0,
                          use_log_scale=use_log_scale)
    
    # Create alive dictionary components plots
    print("\n" + "=" * 80)
    print("Creating Alive Dictionary Components plots...")
    print("=" * 80)
    plot_alive_dictionary_components(data, layers, Path(args.output_dir),
                                   max_mse=args.max_mse, max_l0=args.max_l0,
                                   min_mse=args.min_mse, min_l0=args.min_l0)
    
    # Print summary with filtering
    print_pareto_summary(data, layers, max_mse=args.max_mse, max_l0=args.max_l0,
                        min_mse=args.min_mse, min_l0=args.min_l0)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Plot L0 vs Explained Variance Pareto curves comparing:
- Lagrangian SAE (bandwidth=0.1)
- Lagrangian SAE (bandwidth=0.5)
- BatchTopK
- TopK

Each curve has 4 points for K/target_l0 = 16, 32, 64, 128.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

from settings import settings
from utils.io import load_metrics_from_wandb


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


# Run name patterns - {k} will be replaced with actual K values
RUN_PATTERNS = {
    'lagrangian_bw_0.1': 'lagrangian_target_l0_{k}_alpha_max_1_bandwidth_0.1',
    'lagrangian_bw_0.5': 'lagrangian_target_l0_{k}_alpha_max_1_bandwidth_0.5',
    'batch_topk': 'batch_topk_k_{k}',
    'topk': 'topk_k_{k}',
}

# Project mapping for each SAE type
PROJECT_MAP = {
    'lagrangian_bw_0.1': 'lagrangian-sae/gpt2-small-sweep',
    'lagrangian_bw_0.5': 'lagrangian-sae/gpt2-small-sweep',
    'batch_topk': 'lagrangian-sae/gpt2-small',
    'topk': 'lagrangian-sae/gpt2-small',
}

# K/target_l0 values to plot
K_VALUES = [16, 32, 64, 128]

# Visual styling
COLORS = {
    'lagrangian_bw_0.1': '#d62728',  # Red
    'lagrangian_bw_0.5': '#ff7f0e',  # Orange
    'batch_topk': '#2ca02c',          # Green
    'topk': '#1f77b4',                # Blue
}

MARKERS = {
    'lagrangian_bw_0.1': 'D',  # Diamond
    'lagrangian_bw_0.5': '^',  # Triangle up
    'batch_topk': 's',          # Square
    'topk': 'o',                # Circle
}

LABELS = {
    'lagrangian_bw_0.1': 'Lagrangian (bw=0.1)',
    'lagrangian_bw_0.5': 'Lagrangian (bw=0.5)',
    'batch_topk': 'BatchTopK',
    'topk': 'TopK',
}


def get_run_id_by_name(api: wandb.Api, project: str, run_name: str) -> str | None:
    """Find run ID by exact name match in a project."""
    try:
        runs = api.runs(project, filters={"display_name": run_name})
        runs_list = list(runs)
        if runs_list:
            # Return the most recent run with this name
            return runs_list[0].id
        return None
    except Exception as e:
        print(f"  Error finding run {run_name}: {e}")
        return None


def collect_metrics_data() -> tuple[Dict[str, List[Dict]], List[str]]:
    """
    Collect metrics data for all SAE types from Wandb.
    
    Returns:
        Tuple of (data dict, list of layer names)
    """
    print("Collecting metrics data from Wandb...")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Data structure: {sae_type: [{k, layers: {layer_name: {l0, ev}}}]}
    data: Dict[str, List[Dict]] = {sae_type: [] for sae_type in RUN_PATTERNS.keys()}
    all_layers: set = set()
    
    for sae_type, pattern in RUN_PATTERNS.items():
        project = PROJECT_MAP[sae_type]
        print(f"\nCollecting {LABELS[sae_type]} from {project}...")
        
        for k in K_VALUES:
            run_name = pattern.format(k=k)
            print(f"  Looking for run: {run_name}")
            
            # Find run ID by name
            run_id = get_run_id_by_name(api, project, run_name)
            
            if run_id is None:
                print(f"    WARNING: Run not found!")
                continue
            
            # Load metrics
            metrics = load_metrics_from_wandb(run_id, project)
            
            if metrics is None:
                print(f"    WARNING: No metrics found for run {run_name}")
                continue
            
            # Store per-layer data
            run_data = {
                'k': k,
                'run_name': run_name,
                'layers': {}
            }
            
            for layer_name, layer_metrics in metrics.items():
                all_layers.add(layer_name)
                run_data['layers'][layer_name] = {
                    'l0': layer_metrics.get('sparsity_l0', 0),
                    'explained_variance': layer_metrics.get('explained_variance', 0),
                }
            
            data[sae_type].append(run_data)
            print(f"    Found: L0={list(run_data['layers'].values())[0]['l0']:.2f} (first layer)")
    
    # Sort by K value for each SAE type
    for sae_type in data:
        data[sae_type] = sorted(data[sae_type], key=lambda x: x['k'])
    
    print(f"\nCollected data summary:")
    for sae_type, runs in data.items():
        print(f"  {LABELS[sae_type]}: {len(runs)} runs (K={[r['k'] for r in runs]})")
    print(f"Found {len(all_layers)} layers: {sorted(all_layers)}")
    
    return data, sorted(all_layers)


def plot_pareto_curves(data: Dict[str, List[Dict]], layers: List[str], 
                       output_dir: Path = Path("plots")):
    """
    Create L0 vs Explained Variance Pareto curve plots.
    
    Args:
        data: Dictionary with SAE type data
        layers: List of layer names
        output_dir: Output directory for plots
    """
    output_dir.mkdir(exist_ok=True)
    
    # Create a figure for each layer
    for layer_name in layers:
        print(f"\nPlotting layer: {layer_name}")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot each SAE type
        for sae_type in ['topk', 'batch_topk', 'lagrangian_bw_0.5', 'lagrangian_bw_0.1']:
            if not data.get(sae_type):
                continue
            
            # Extract layer-specific data
            l0_values = []
            ev_values = []
            k_labels = []
            
            for run_data in data[sae_type]:
                if layer_name in run_data['layers']:
                    l0 = run_data['layers'][layer_name]['l0']
                    ev = run_data['layers'][layer_name]['explained_variance']
                    l0_values.append(l0)
                    ev_values.append(ev)
                    k_labels.append(run_data['k'])
            
            if not l0_values:
                continue
            
            l0_values = np.array(l0_values)
            ev_values = np.array(ev_values)
            
            # Sort by L0 for line plotting
            sort_idx = np.argsort(l0_values)
            l0_sorted = l0_values[sort_idx]
            ev_sorted = ev_values[sort_idx]
            k_sorted = [k_labels[i] for i in sort_idx]
            
            # Plot points and lines
            ax.scatter(l0_sorted, ev_sorted,
                      color=COLORS[sae_type], marker=MARKERS[sae_type],
                      s=150, label=LABELS[sae_type], zorder=3, alpha=0.8)
            ax.plot(l0_sorted, ev_sorted,
                   color=COLORS[sae_type], linewidth=2.5, alpha=0.7, zorder=2)
            
            # Add K labels to points
            for x, y, k in zip(l0_sorted, ev_sorted, k_sorted):
                ax.annotate(f'{k}', (x, y),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, alpha=0.7, color=COLORS[sae_type],
                           fontweight='bold')
        
        # Styling
        ax.set_xlabel('L0 Sparsity', fontsize=20, labelpad=10)
        ax.set_ylabel('Explained Variance → (better)', fontsize=20, labelpad=10)
        ax.set_title(f'L0 vs Explained Variance - {layer_name}', fontsize=24, pad=15)
        ax.legend(loc='lower right', fontsize=16, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        layer_display_name = layer_name.replace('.', '_')
        output_path_png = output_dir / f"l0_vs_ev_comparison_{layer_display_name}.png"
        output_path_svg = output_dir / f"l0_vs_ev_comparison_{layer_display_name}.svg"
        
        plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        print(f"  Saved: {output_path_png}")
        
        plt.close()


def print_summary(data: Dict[str, List[Dict]], layers: List[str]):
    """Print a summary table of the collected data."""
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    
    for layer_name in layers:
        print(f"\n{layer_name}:")
        print(f"  {'SAE Type':<25} {'K':<6} {'L0':<10} {'EV':<10}")
        print(f"  {'-'*55}")
        
        for sae_type in ['topk', 'batch_topk', 'lagrangian_bw_0.5', 'lagrangian_bw_0.1']:
            if not data.get(sae_type):
                continue
            
            for run_data in data[sae_type]:
                if layer_name in run_data['layers']:
                    l0 = run_data['layers'][layer_name]['l0']
                    ev = run_data['layers'][layer_name]['explained_variance']
                    print(f"  {LABELS[sae_type]:<25} {run_data['k']:<6} {l0:<10.2f} {ev:<10.4f}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot L0 vs Explained Variance Pareto curves for Lagrangian vs baselines"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, don't generate plots"
    )
    
    args = parser.parse_args()
    
    # Collect data
    print("=" * 80)
    print("Collecting data from Wandb...")
    print("=" * 80)
    data, layers = collect_metrics_data()
    
    # Print summary
    print_summary(data, layers)
    
    if not args.summary_only:
        # Create plots
        print("\n" + "=" * 80)
        print("Creating Pareto curve plots...")
        print("=" * 80)
        plot_pareto_curves(data, layers, Path(args.output_dir))
        
        print("\n" + "=" * 80)
        print("Done!")
        print("=" * 80)


if __name__ == "__main__":
    main()


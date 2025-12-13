
from typing import Any
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def create_pareto_plots(all_run_metrics: list[dict[str, Any]]) -> None:
    """
    Create pareto plots showing trade-offs between sparsity and other metrics, with separate curves per layer.
    
    Args:
        all_run_metrics: List of dictionaries containing metrics for each run
    """
    
    
    # Prepare data for plotting, grouped by layer
    layer_data = defaultdict(list)
    
    for run_metric in all_run_metrics:
        run_name = run_metric['run_name']
        metrics = run_metric['metrics']
        config = run_metric['config']
        
        # Extract sparsity coefficient from config
        try:
            sparsity_coeff = config['loss']['sparsity']['coeff']
        except (KeyError, TypeError):
            sparsity_coeff = 'N/A'
        
        # Extract metrics for each SAE position for this run
        for sae_pos, pos_metrics in metrics.items():
            # Skip if metrics weren't computed (loading existing data case)
            if isinstance(pos_metrics.get('sparsity_l0'), str):
                continue
                
            layer_data[sae_pos].append({
                'run_name': run_name,
                'sae_position': sae_pos,
                'sparsity_l0': pos_metrics['sparsity_l0'],
                'mse': pos_metrics['mse'],
                'explained_variance': pos_metrics['explained_variance'],
                'alive_dict_proportion': pos_metrics['alive_dict_components_proportion'],
                'sparsity_coeff': sparsity_coeff
            })
    
    if not layer_data:
        print("No valid metrics found for plotting.")
        return
    
    # Get unique layers and sort them
    unique_layers = sorted(layer_data.keys())
    n_layers = len(unique_layers)
    
    print(f"Creating pareto plots for {n_layers} layers: {unique_layers}")
    
    # Create figure with subplots (3 metric types × number of layers)
    fig, axes = plt.subplots(3, n_layers, figsize=(6*n_layers, 18))
    
    # Handle case where there's only one layer
    if n_layers == 1:
        axes = axes.reshape(3, 1)
    
    output_dir = Path("pareto_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Plot for each layer
    for layer_idx, layer_name in enumerate(unique_layers):
        plot_data = layer_data[layer_name]
        
        # Convert to arrays for easier plotting
        sparsity = np.array([d['sparsity_l0'] for d in plot_data])
        mse = np.array([d['mse'] for d in plot_data])
        explained_variance = np.array([d['explained_variance'] for d in plot_data])
        alive_dict_proportion = np.array([d['alive_dict_proportion'] for d in plot_data])
        run_names = [d['run_name'] for d in plot_data]
        sparsity_coeffs = [d['sparsity_coeff'] for d in plot_data]
        
        # Color by run type (differentiate ReLU vs Bayesian if applicable)
        colors = []
        for name in run_names:
            if 'bayesian' in name.lower() or 'variational' in name.lower():
                colors.append('red')
            elif 'relu' in name.lower():
                colors.append('blue')
            else:
                colors.append('gray')
        
        # Plot 1: Sparsity vs MSE
        axes[0, layer_idx].scatter(sparsity, mse, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, mse, sparsity_coeffs)):
            if coeff != 'N/A':
                axes[0, layer_idx].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8, alpha=0.7)
        axes[0, layer_idx].set_xlabel('L0 Sparsity')
        axes[0, layer_idx].set_ylabel('MSE (Reconstruction Loss)')
        axes[0, layer_idx].set_title(f'{layer_name}: Sparsity vs MSE')
        axes[0, layer_idx].grid(True, alpha=0.3)
        
        # Plot 2: Sparsity vs Explained Variance
        axes[1, layer_idx].scatter(sparsity, explained_variance, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, explained_variance, sparsity_coeffs)):
            if coeff != 'N/A':
                axes[1, layer_idx].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8, alpha=0.7)
        axes[1, layer_idx].set_xlabel('L0 Sparsity')
        axes[1, layer_idx].set_ylabel('Explained Variance')
        axes[1, layer_idx].set_title(f'{layer_name}: Sparsity vs Explained Variance')
        axes[1, layer_idx].grid(True, alpha=0.3)
        
        # Plot 3: Sparsity vs Alive Dictionary Proportion
        axes[2, layer_idx].scatter(sparsity, alive_dict_proportion, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, alive_dict_proportion, sparsity_coeffs)):
            if coeff != 'N/A':
                axes[2, layer_idx].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8, alpha=0.7)
        axes[2, layer_idx].set_xlabel('L0 Sparsity')
        axes[2, layer_idx].set_ylabel('Alive Dictionary Elements Proportion')
        axes[2, layer_idx].set_title(f'{layer_name}: Sparsity vs Alive Dict Elements')
        axes[2, layer_idx].grid(True, alpha=0.3)
        
        # Add legend to the first plot of each row
        if layer_idx == 0:
            unique_colors = list(set(colors))
            if len(unique_colors) > 1:
                legend_elements = []
                if 'red' in unique_colors:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Bayesian SAE'))
                if 'blue' in unique_colors:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='ReLU SAE'))
                if 'gray' in unique_colors:
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other'))
                
                if legend_elements:
                    axes[1, layer_idx].legend(handles=legend_elements, loc='upper right')
        
        # Print layer-specific statistics
        valid_coeffs = [c for c in sparsity_coeffs if c != 'N/A']
        print(f"\n{layer_name} statistics ({len(plot_data)} data points):")
        print(f"  Sparsity range: {sparsity.min():.4f} - {sparsity.max():.4f}")
        print(f"  MSE range: {mse.min():.4f} - {mse.max():.4f}")
        print(f"  Explained Variance range: {explained_variance.min():.4f} - {explained_variance.max():.4f}")
        print(f"  Alive Dict Proportion range: {alive_dict_proportion.min():.4f} - {alive_dict_proportion.max():.4f}")
        if valid_coeffs:
            print(f"  Sparsity coefficient range: {min(valid_coeffs):.2e} - {max(valid_coeffs):.2e}")
        else:
            print(f"  Sparsity coefficient: N/A")
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / "sparsity_pareto_plots_by_layer.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "sparsity_pareto_plots_by_layer.svg", bbox_inches='tight')
    
    # Also create individual plots for each layer for better readability
    for layer_idx, layer_name in enumerate(unique_layers):
        plot_data = layer_data[layer_name]
        
        # Convert to arrays for easier plotting
        sparsity = np.array([d['sparsity_l0'] for d in plot_data])
        mse = np.array([d['mse'] for d in plot_data])
        explained_variance = np.array([d['explained_variance'] for d in plot_data])
        alive_dict_proportion = np.array([d['alive_dict_proportion'] for d in plot_data])
        run_names = [d['run_name'] for d in plot_data]
        sparsity_coeffs = [d['sparsity_coeff'] for d in plot_data]
        
        # Color by run type
        colors = []
        for name in run_names:
            if 'bayesian' in name.lower() or 'variational' in name.lower():
                colors.append('red')
            elif 'relu' in name.lower():
                colors.append('blue')
            else:
                colors.append('gray')
        
        # Create individual figure for this layer
        fig_individual, axes_individual = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Sparsity vs MSE
        axes_individual[0].scatter(sparsity, mse, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, mse, sparsity_coeffs)):
            if coeff != 'N/A':
                axes_individual[0].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=10, alpha=0.8)
        axes_individual[0].set_xlabel('L0 Sparsity')
        axes_individual[0].set_ylabel('MSE (Reconstruction Loss)')
        axes_individual[0].set_title(f'{layer_name}: Sparsity vs MSE')
        axes_individual[0].grid(True, alpha=0.3)
        
        # Plot 2: Sparsity vs Explained Variance
        axes_individual[1].scatter(sparsity, explained_variance, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, explained_variance, sparsity_coeffs)):
            if coeff != 'N/A':
                axes_individual[1].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=10, alpha=0.8)
        axes_individual[1].set_xlabel('L0 Sparsity')
        axes_individual[1].set_ylabel('Explained Variance')
        axes_individual[1].set_title(f'{layer_name}: Sparsity vs Explained Variance')
        axes_individual[1].grid(True, alpha=0.3)
        
        # Plot 3: Sparsity vs Alive Dictionary Proportion
        axes_individual[2].scatter(sparsity, alive_dict_proportion, alpha=0.7, s=50, c=colors)
        # Add sparsity coefficient annotations
        for i, (x, y, coeff) in enumerate(zip(sparsity, alive_dict_proportion, sparsity_coeffs)):
            if coeff != 'N/A':
                axes_individual[2].annotate(f'{coeff}', (x, y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=10, alpha=0.8)
        axes_individual[2].set_xlabel('L0 Sparsity')
        axes_individual[2].set_ylabel('Alive Dictionary Elements Proportion')
        axes_individual[2].set_title(f'{layer_name}: Sparsity vs Alive Dict Elements')
        axes_individual[2].grid(True, alpha=0.3)
        
        # Add legend
        unique_colors = list(set(colors))
        if len(unique_colors) > 1:
            legend_elements = []
            if 'red' in unique_colors:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Bayesian SAE'))
            if 'blue' in unique_colors:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='ReLU SAE'))
            if 'gray' in unique_colors:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other'))
            
            if legend_elements:
                axes_individual[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save individual layer plot
        safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
        plt.savefig(output_dir / f"pareto_{safe_layer_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f"pareto_{safe_layer_name}.svg", bbox_inches='tight')
        plt.close(fig_individual)
    
    total_points = sum(len(layer_data[layer]) for layer in unique_layers)
    unique_runs = set()
    for layer_points in layer_data.values():
        for point in layer_points:
            unique_runs.add(point['run_name'])
    
    print(f"\nSaved pareto plots to {output_dir}/")
    print(f"  - Combined: sparsity_pareto_plots_by_layer.png|svg")
    print(f"  - Individual: pareto_{{layer_name}}.png|svg for each layer")
    print(f"Plotted {total_points} total data points from {len(unique_runs)} unique runs across {n_layers} layers")
    
    plt.show()

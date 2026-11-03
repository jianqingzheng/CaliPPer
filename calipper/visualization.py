'''
Visualization module for generalization evaluation
Jianqing Zheng
2025.02.06
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Optional, Tuple


def plot_generalization_curve(distance_values: List[float],
                               performance_values: List[float],
                               sample_counts: List[int],
                               distance_metric: str,
                               performance_metric: str,
                               model_name: str,
                               dataset_name: str,
                               output_dir: str = 'figures',
                               show: bool = False,
                               figsize: Tuple[int, int] = (10, 6)) -> str:
    """
    Enhanced generalization curve with sample counts and styling.

    Parameters:
    -----------
    distance_values : list of float
        Mean distance values per bin (x-axis)
    performance_values : list of float
        Performance scores per bin (y-axis)
    sample_counts : list of int
        Number of samples in each bin
    distance_metric : str
        Name of distance metric (e.g., 'seq_edit_dist')
    performance_metric : str
        Name of performance metric (e.g., 'aucroc')
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset
    output_dir : str
        Directory to save figure
    show : bool
        Whether to display the plot
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    str : Path to saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the curve with markers
    ax.plot(distance_values, performance_values, 'o-',
            linewidth=2, markersize=8, label=f'{model_name}')

    # Add sample count annotations
    for i, (x, y, n) in enumerate(zip(distance_values, performance_values, sample_counts)):
        ax.annotate(f'n={n}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8, alpha=0.7)

    # Styling
    ax.set_xlabel(f'Distribution Distance ({distance_metric})', fontsize=12)
    ax.set_ylabel(f'Performance ({performance_metric})', fontsize=12)
    ax.set_title(f'Generalization Analysis: {model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    output_file = os.path.join(output_dir, f'generalization_{model_name}_{dataset_name}_{distance_metric}_{performance_metric}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return output_file


def plot_multi_metric_comparison(eval_results_dict: Dict[str, List],
                                  distance_metric: str,
                                  model_name: str,
                                  dataset_name: str,
                                  metrics: List[str] = ['acc', 'f1', 'aucroc'],
                                  output_dir: str = 'figures',
                                  figsize: Tuple[int, int] = (15, 5)) -> str:
    """
    Create subplots comparing different performance metrics.

    Parameters:
    -----------
    eval_results_dict : dict
        Dictionary mapping metric names to [perf_values, dist_values, counts]
        Example: {'acc': [[0.8, 0.7, ...], [0.1, 0.2, ...], [100, 120, ...]]}
    distance_metric : str
        Name of distance metric
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset
    metrics : list of str
        List of performance metrics to compare
    output_dir : str
        Directory to save figure
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    str : Path to saved figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        if metric not in eval_results_dict:
            print(f"Warning: Metric '{metric}' not found in results dictionary")
            continue

        perf_values, dist_values, counts = eval_results_dict[metric]
        ax = axes[idx]

        # Plot
        ax.plot(dist_values, perf_values, 'o-', linewidth=2, markersize=8)

        # Styling
        ax.set_xlabel(f'Distance ({distance_metric})', fontsize=10)
        ax.set_ylabel(f'{metric.upper()}', fontsize=10)
        ax.set_title(f'{metric.upper()} vs Distance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add sample counts
        for x, y, n in zip(dist_values, perf_values, counts):
            ax.annotate(f'{n}', (x, y), textcoords="offset points",
                       xytext=(0, 5), ha='center', fontsize=7, alpha=0.6)

    plt.suptitle(f'Multi-Metric Comparison: {model_name} on {dataset_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    metrics_str = '_'.join(metrics)
    output_file = os.path.join(output_dir, f'multi_metric_{model_name}_{dataset_name}_{distance_metric}_{metrics_str}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_multi_distance_comparison(eval_results_dict: Dict[str, List],
                                    performance_metric: str,
                                    model_name: str,
                                    dataset_name: str,
                                    distance_metrics: List[str] = ['seq_edit_dist', 'struct_embed_dist'],
                                    output_dir: str = 'figures',
                                    figsize: Tuple[int, int] = (10, 6)) -> str:
    """
    Overlay curves for different distance metrics on same plot.

    Parameters:
    -----------
    eval_results_dict : dict
        Dictionary mapping distance metric names to [perf_values, dist_values, counts]
    performance_metric : str
        Name of performance metric
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset
    distance_metrics : list of str
        List of distance metrics to compare
    output_dir : str
        Directory to save figure
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    str : Path to saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(distance_metrics)))

    for idx, dist_metric in enumerate(distance_metrics):
        if dist_metric not in eval_results_dict:
            print(f"Warning: Distance metric '{dist_metric}' not found in results dictionary")
            continue

        perf_values, dist_values, counts = eval_results_dict[dist_metric]

        # Plot with different colors
        ax.plot(dist_values, perf_values, 'o-',
                linewidth=2, markersize=8, label=dist_metric, color=colors[idx])

    # Styling
    ax.set_xlabel('Distribution Distance (normalized)', fontsize=12)
    ax.set_ylabel(f'Performance ({performance_metric})', fontsize=12)
    ax.set_title(f'Distance Metric Comparison: {model_name} on {dataset_name}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    dist_str = '_'.join([d.replace('_', '') for d in distance_metrics])
    output_file = os.path.join(output_dir, f'multi_distance_{model_name}_{dataset_name}_{performance_metric}_{dist_str}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_multi_model_comparison(eval_results_dict: Dict[str, List],
                                 distance_metric: str,
                                 performance_metric: str,
                                 dataset_name: str,
                                 models: List[str],
                                 output_dir: str = 'figures',
                                 figsize: Tuple[int, int] = (10, 6)) -> str:
    """
    Overlay curves for different models on same plot.

    Parameters:
    -----------
    eval_results_dict : dict
        Dictionary mapping model names to [perf_values, dist_values, counts]
    distance_metric : str
        Name of distance metric
    performance_metric : str
        Name of performance metric
    dataset_name : str
        Name of the dataset
    models : list of str
        List of model names to compare
    output_dir : str
        Directory to save figure
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    str : Path to saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, model in enumerate(models):
        if model not in eval_results_dict:
            print(f"Warning: Model '{model}' not found in results dictionary")
            continue

        perf_values, dist_values, counts = eval_results_dict[model]

        # Plot with different colors and markers
        marker = markers[idx % len(markers)]
        ax.plot(dist_values, perf_values, f'{marker}-',
                linewidth=2, markersize=8, label=model, color=colors[idx])

    # Styling
    ax.set_xlabel(f'Distribution Distance ({distance_metric})', fontsize=12)
    ax.set_ylabel(f'Performance ({performance_metric})', fontsize=12)
    ax.set_title(f'Model Comparison on {dataset_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best')

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    models_str = '_'.join(models)
    output_file = os.path.join(output_dir, f'multi_model_{dataset_name}_{distance_metric}_{performance_metric}_{models_str}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_heatmap(performance_matrix: np.ndarray,
                 row_labels: List[str],
                 col_labels: List[str],
                 title: str,
                 xlabel: str = 'Distance Bins',
                 ylabel: str = 'Models/Metrics',
                 output_dir: str = 'figures',
                 output_name: str = 'heatmap',
                 figsize: Tuple[int, int] = (12, 8),
                 cmap: str = 'RdYlGn') -> str:
    """
    Create a 2D heatmap of performance values.

    Parameters:
    -----------
    performance_matrix : np.ndarray
        2D array of performance values (rows: models/metrics, cols: distance bins)
    row_labels : list of str
        Labels for rows (e.g., model names or metric names)
    col_labels : list of str
        Labels for columns (e.g., distance bin labels)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    output_dir : str
        Directory to save figure
    output_name : str
        Output filename (without extension)
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name

    Returns:
    --------
    str : Path to saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(performance_matrix, cmap=cmap, aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance', rotation=270, labelpad=20, fontsize=10)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if not np.isnan(performance_matrix[i, j]):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{output_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def save_all_visualizations(results: Dict,
                            output_dir: str = 'figures',
                            model_name: str = None,
                            dataset_name: str = None) -> List[str]:
    """
    Generate all visualization types for a complete results dictionary.

    Parameters:
    -----------
    results : dict
        Complete results dictionary with structure:
        {
            'model_name': str,
            'dataset_name': str,
            'distance_metric': str,
            'performance_metric': str,
            'eval_data': [perf_values, dist_values, counts]
        }
    output_dir : str
        Directory to save figures
    model_name : str, optional
        Override model name from results dict
    dataset_name : str, optional
        Override dataset name from results dict

    Returns:
    --------
    list of str : List of paths to generated figures
    """
    generated_files = []

    # Extract information
    model = model_name or results.get('model_name', 'unknown_model')
    dataset = dataset_name or results.get('dataset_name', 'unknown_dataset')
    dist_metric = results.get('distance_metric', 'unknown_distance')
    perf_metric = results.get('performance_metric', 'unknown_performance')
    eval_data = results.get('eval_data', [[], [], []])

    perf_values, dist_values, counts = eval_data

    # Generate single curve plot
    if perf_values and dist_values and counts:
        try:
            fig_path = plot_generalization_curve(
                dist_values, perf_values, counts,
                dist_metric, perf_metric,
                model, dataset,
                output_dir=output_dir
            )
            generated_files.append(fig_path)
            print(f"Generated: {fig_path}")
        except Exception as e:
            print(f"Error generating generalization curve: {e}")

    return generated_files


if __name__ == '__main__':
    # Example usage
    print("Visualization module loaded successfully")

    # Test with dummy data
    distance_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    performance_values = [0.95, 0.92, 0.88, 0.83, 0.78, 0.72, 0.65, 0.58]
    sample_counts = [150, 145, 148, 152, 149, 151, 147, 158]

    output_file = plot_generalization_curve(
        distance_values, performance_values, sample_counts,
        'seq_edit_dist', 'aucroc',
        'TestModel', 'TestDataset',
        output_dir='figures',
        show=False
    )
    print(f"Test figure saved to: {output_file}")

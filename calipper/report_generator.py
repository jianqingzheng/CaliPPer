'''
Comprehensive report generation module
Creates publication-ready reports with embedded figures and analysis
Jianqing Zheng
2025.02.06
'''

import os
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .visualization import (
    plot_generalization_curve,
    plot_multi_metric_comparison,
    plot_multi_distance_comparison
)
from .pattern_analysis import (
    select_best_fit,
    classify_pattern,
    identify_reliable_region,
    compute_degradation_rate
)
from .statistical_analysis import (
    comprehensive_statistical_report,
    compute_correlation
)
from .utils import df_to_markdown


def generate_figure_gallery(eval_results: Dict[str, List],
                            model_name: str,
                            dataset_name: str,
                            output_dir: str) -> List[str]:
    """
    Generate all visualization figures for a model-dataset combination.

    Parameters:
    -----------
    eval_results : dict
        Results dict with keys as metric names, values as [perf, dist, counts]
    model_name : str
        Model name
    dataset_name : str
        Dataset name
    output_dir : str
        Output directory for figures

    Returns:
    --------
    list : Paths to generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    figure_paths = []

    # Extract distance and performance metrics
    distance_metrics = set()
    performance_metrics = set()

    for key in eval_results.keys():
        parts = key.split('_')
        if len(parts) >= 2:
            dist = '_'.join(parts[:-1])
            perf = parts[-1]
            distance_metrics.add(dist)
            performance_metrics.add(perf)

    # 1. Individual curves for each combination
    for metric_key, (perf_vals, dist_vals, counts) in eval_results.items():
        parts = metric_key.split('_')
        dist_metric = '_'.join(parts[:-1])
        perf_metric = parts[-1]

        fig_path = plot_generalization_curve(
            distance_values=dist_vals,
            performance_values=perf_vals,
            sample_counts=counts,
            distance_metric=dist_metric,
            performance_metric=perf_metric,
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            show=False
        )
        figure_paths.append(fig_path)

    # 2. Multi-metric comparison (if multiple performance metrics)
    if len(performance_metrics) > 1:
        for dist_metric in distance_metrics:
            # Gather results for this distance metric
            metric_results = {}
            for perf_metric in performance_metrics:
                key = f"{dist_metric}_{perf_metric}"
                if key in eval_results:
                    metric_results[perf_metric] = eval_results[key]

            if len(metric_results) > 1:
                fig_path = plot_multi_metric_comparison(
                    eval_results_dict=metric_results,
                    distance_metric=dist_metric,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    metrics=list(metric_results.keys()),
                    output_dir=output_dir
                )
                figure_paths.append(fig_path)

    # 3. Multi-distance comparison (if multiple distance metrics)
    if len(distance_metrics) > 1:
        for perf_metric in performance_metrics:
            # Gather results for this performance metric
            distance_results = {}
            for dist_metric in distance_metrics:
                key = f"{dist_metric}_{perf_metric}"
                if key in eval_results:
                    distance_results[dist_metric] = eval_results[key]

            if len(distance_results) > 1:
                fig_path = plot_multi_distance_comparison(
                    eval_results_dict=distance_results,
                    performance_metric=perf_metric,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    distance_metrics=list(distance_results.keys()),
                    output_dir=output_dir
                )
                figure_paths.append(fig_path)

    return figure_paths


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string for HTML embedding.

    Parameters:
    -----------
    image_path : str
        Path to image file

    Returns:
    --------
    str : Base64 encoded image string
    """
    with open(image_path, 'rb') as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')


def generate_html_report(markdown_text: str,
                         figure_paths: List[str],
                         title: str,
                         output_file: str) -> str:
    """
    Generate HTML report from markdown with embedded figures.

    Parameters:
    -----------
    markdown_text : str
        Markdown report text
    figure_paths : list
        Paths to figure files
    title : str
        Report title
    output_file : str
        Output HTML file path

    Returns:
    --------
    str : HTML file path
    """
    # Convert markdown to HTML-friendly format
    html_content = markdown_text.replace('\n', '<br>\n')

    # Embed figures as base64
    embedded_figures = []
    for fig_path in figure_paths:
        if os.path.exists(fig_path):
            img_base64 = image_to_base64(fig_path)
            fig_name = os.path.basename(fig_path)
            embedded_figures.append(f'''
    <div class="figure">
        <img src="data:image/png;base64,{img_base64}" alt="{fig_name}">
        <p class="caption">{fig_name}</p>
    </div>
    ''')

    # HTML template
    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .figure {{
            margin: 20px 0;
            text-align: center;
            page-break-inside: avoid;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
        }}
        .caption {{
            font-style: italic;
            color: #7f8c8d;
            margin-top: 8px;
            font-size: 0.9em;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        .highlight {{
            background-color: #ffffcc;
            padding: 2px 4px;
            border-radius: 2px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        @media print {{
            body {{
                background-color: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="metadata">
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Figures:</strong> {len(figure_paths)} visualizations included
        </div>

        <div class="content">
            {html_content}
        </div>

        <h2>Figures</h2>
        {''.join(embedded_figures)}

        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
            Generated by General Evaluation Framework
        </footer>
    </div>
</body>
</html>
'''

    # Save HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_file


def generate_comprehensive_report(distance_values: List[float],
                                  performance_values: List[float],
                                  sample_counts: List[int],
                                  distance_metric: str,
                                  performance_metric: str,
                                  model_name: str,
                                  dataset_name: str,
                                  output_dir: str = 'reports',
                                  include_html: bool = True) -> Dict[str, str]:
    """
    Generate comprehensive evaluation report with all analyses.

    Parameters:
    -----------
    distance_values : list
        Distance values per bin
    performance_values : list
        Performance values per bin
    sample_counts : list
        Sample counts per bin
    distance_metric : str
        Distance metric name
    performance_metric : str
        Performance metric name
    model_name : str
        Model name
    dataset_name : str
        Dataset name
    output_dir : str
        Output directory
    include_html : bool
        Whether to generate HTML version

    Returns:
    --------
    dict : Paths to generated reports
    """
    os.makedirs(output_dir, exist_ok=True)

    report_lines = []

    # Header
    report_lines.append(f"# Generalization Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Model:** {model_name}")
    report_lines.append(f"**Dataset:** {dataset_name}")
    report_lines.append(f"**Distance Metric:** {distance_metric}")
    report_lines.append(f"**Performance Metric:** {performance_metric}")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")

    mean_perf = np.mean(performance_values)
    perf_range = np.max(performance_values) - np.min(performance_values)
    total_samples = np.sum(sample_counts)

    report_lines.append(f"- **Total Samples:** {total_samples}")
    report_lines.append(f"- **Distance Bins:** {len(distance_values)}")
    report_lines.append(f"- **Mean Performance:** {mean_perf:.4f}")
    report_lines.append(f"- **Performance Range:** {perf_range:.4f}")
    report_lines.append("")

    # Pattern Analysis
    report_lines.append("## Pattern Analysis")
    report_lines.append("")

    pattern_type, confidence = classify_pattern(distance_values, performance_values)
    best_model, best_fit = select_best_fit(distance_values, performance_values)
    deg_rate = compute_degradation_rate(distance_values, performance_values)
    max_reliable_dist, (min_dist, max_dist) = identify_reliable_region(
        distance_values, performance_values, threshold=0.8
    )

    report_lines.append(f"### Degradation Pattern")
    report_lines.append(f"- **Pattern Type:** {pattern_type}")
    report_lines.append(f"- **Classification Confidence:** {confidence:.3f}")
    report_lines.append(f"- **Best Fit Model:** {best_model}")
    report_lines.append(f"- **R² Score:** {best_fit['r2']:.4f}")
    report_lines.append(f"- **RMSE:** {best_fit['rmse']:.4f}")
    report_lines.append("")

    report_lines.append(f"### Degradation Metrics")
    report_lines.append(f"- **Mean Degradation Rate:** {deg_rate['mean_rate']:.4f}")
    report_lines.append(f"- **Total Performance Change:** {deg_rate['total_change']:.4f}")
    report_lines.append(f"- **Relative Change:** {deg_rate['relative_change']:.2%}")
    report_lines.append("")

    report_lines.append(f"### Reliable Operating Region")
    report_lines.append(f"- **Maximum Reliable Distance:** {max_reliable_dist:.4f}")
    report_lines.append(f"  (Distance range where performance ≥ 80% of maximum)")
    report_lines.append("")

    # Statistical Analysis
    report_lines.append("## Statistical Analysis")
    report_lines.append("")

    stats_report = comprehensive_statistical_report(
        distance_values, performance_values, sample_counts
    )

    # Correlation
    report_lines.append("### Correlation Analysis")
    pearson = stats_report['pearson']
    spearman = stats_report['spearman']
    report_lines.append(f"- **Pearson r:** {pearson['correlation']:.4f} (p={pearson['p_value']:.4f})")
    report_lines.append(f"  - {pearson['interpretation']}")
    report_lines.append(f"  - {'Statistically significant' if pearson['significant'] else 'Not significant'}")
    report_lines.append(f"- **Spearman ρ:** {spearman['correlation']:.4f} (p={spearman['p_value']:.4f})")
    report_lines.append(f"  - {spearman['interpretation']}")
    report_lines.append(f"  - {'Statistically significant' if spearman['significant'] else 'Not significant'}")
    report_lines.append("")

    # Monotonicity
    report_lines.append("### Monotonicity Test")
    mono = stats_report['monotonicity']
    report_lines.append(f"- **Monotonic Degradation:** {mono['is_monotonic']}")
    report_lines.append(f"- **Monotonicity Score:** {mono['monotonicity_score']:.4f}")
    report_lines.append(f"- **Violations:** {mono['violations']} out of {mono['total_transitions']} transitions")
    if mono['violations'] > 0:
        report_lines.append(f"- **Maximum Violation:** {mono['max_violation']:.4f}")
    report_lines.append("")

    # Group comparison
    if stats_report['group_comparison']:
        report_lines.append("### Performance Degradation Significance")
        comp = stats_report['group_comparison']
        report_lines.append(f"- **Test:** {comp['test']}")
        report_lines.append(f"- **Low Distance Performance:** {comp['group1_mean']:.4f}")
        report_lines.append(f"- **High Distance Performance:** {comp['group2_mean']:.4f}")
        report_lines.append(f"- **Mean Difference:** {comp['mean_difference']:.4f}")
        report_lines.append(f"- **p-value:** {comp['p_value']:.4f}")
        report_lines.append(f"- **Result:** {comp['interpretation']}")

        if stats_report['effect_size']:
            cohens_d = stats_report['effect_size']['cohens_d']
            if not np.isnan(cohens_d):
                magnitude = 'small' if abs(cohens_d) < 0.2 else ('medium' if abs(cohens_d) < 0.8 else 'large')
                report_lines.append(f"- **Effect Size (Cohen's d):** {cohens_d:.4f} ({magnitude})")
        report_lines.append("")

    # Performance drop thresholds
    report_lines.append("### Performance Drop Thresholds")
    thresh_10 = stats_report['threshold_10pct']
    thresh_20 = stats_report['threshold_20pct']
    report_lines.append(f"- **10% Performance Drop:**")
    report_lines.append(f"  - Distance threshold: {thresh_10['threshold_distance']:.4f}")
    report_lines.append(f"  - Initial performance: {thresh_10['initial_performance']:.4f}")
    report_lines.append(f"  - Performance at threshold: {thresh_10['threshold_performance']:.4f}")
    report_lines.append(f"- **20% Performance Drop:**")
    report_lines.append(f"  - Distance threshold: {thresh_20['threshold_distance']:.4f}")
    report_lines.append(f"  - Initial performance: {thresh_20['initial_performance']:.4f}")
    report_lines.append(f"  - Performance at threshold: {thresh_20['threshold_performance']:.4f}")
    report_lines.append("")

    # Data Table
    report_lines.append("## Evaluation Data")
    report_lines.append("")
    data_df = pd.DataFrame({
        'Bin': range(1, len(distance_values) + 1),
        'Distance': [f'{d:.4f}' for d in distance_values],
        'Performance': [f'{p:.4f}' for p in performance_values],
        'Sample Count': sample_counts
    })
    report_lines.append(df_to_markdown(data_df))
    report_lines.append("")

    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")

    if pattern_type == 'plateau':
        report_lines.append("✓ **Good Generalization:** Model maintains stable performance across distance range.")
        report_lines.append("  - Consider this configuration for production deployment.")
    elif pattern_type == 'linear':
        report_lines.append("⚠ **Moderate Generalization:** Performance degrades linearly with distance.")
        report_lines.append(f"  - Reliable up to distance {max_reliable_dist:.4f}")
        report_lines.append("  - Consider retraining with more diverse data.")
    elif pattern_type == 'exponential':
        report_lines.append("⚠ **Rapid Degradation:** Performance drops quickly with distance.")
        report_lines.append(f"  - Use caution beyond distance {max_reliable_dist:.4f}")
        report_lines.append("  - Strong recommendation: improve model robustness.")
    elif pattern_type == 'step':
        report_lines.append("⚠ **Sudden Performance Drop:** Model shows step-like degradation.")
        report_lines.append(f"  - Identify the critical distance threshold around {max_reliable_dist:.4f}")
        report_lines.append("  - Investigate causes of sudden performance change.")

    report_lines.append("")

    # Save markdown report
    markdown_file = os.path.join(
        output_dir,
        f'evaluation_report_{model_name}_{dataset_name}_{distance_metric}_{performance_metric}.md'
    )
    markdown_text = '\n'.join(report_lines)
    with open(markdown_file, 'w') as f:
        f.write(markdown_text)

    results = {'markdown': markdown_file}

    # Generate figures
    figures_dir = os.path.join(output_dir, 'figures')
    fig_path = plot_generalization_curve(
        distance_values=distance_values,
        performance_values=performance_values,
        sample_counts=sample_counts,
        distance_metric=distance_metric,
        performance_metric=performance_metric,
        model_name=model_name,
        dataset_name=dataset_name,
        output_dir=figures_dir,
        show=False
    )
    figure_paths = [fig_path]

    # Generate HTML report if requested
    if include_html:
        html_file = os.path.join(
            output_dir,
            f'evaluation_report_{model_name}_{dataset_name}_{distance_metric}_{performance_metric}.html'
        )
        title = f"Generalization Evaluation: {model_name} on {dataset_name}"
        html_path = generate_html_report(
            markdown_text=markdown_text,
            figure_paths=figure_paths,
            title=title,
            output_file=html_file
        )
        results['html'] = html_path

    results['figures'] = figure_paths

    return results


if __name__ == '__main__':
    print("Report generation module loaded successfully")
    print("Use generate_comprehensive_report() to create publication-ready reports")

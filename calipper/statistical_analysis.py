'''
Statistical analysis module for generalization evaluation
Correlation, significance testing, and confidence intervals
Jianqing Zheng
2025.02.06
'''

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, ttest_ind, mannwhitneyu
from sklearn.utils import resample
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def compute_correlation(x: List[float], y: List[float],
                        method: str = 'pearson') -> Dict[str, float]:
    """
    Compute correlation between distance and performance.

    Parameters:
    -----------
    x : list of float
        Distance values
    y : list of float
        Performance values
    method : str
        Correlation method: 'pearson' or 'spearman'

    Returns:
    --------
    dict : Dictionary with:
        - 'correlation': Correlation coefficient
        - 'p_value': Statistical significance (p-value)
        - 'method': Correlation method used
        - 'significant': Boolean indicating if p < 0.05
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 3:
        return {
            'correlation': np.nan,
            'p_value': 1.0,
            'method': method,
            'significant': False,
            'interpretation': 'Insufficient data points'
        }

    if method == 'pearson':
        corr, p_value = pearsonr(x, y)
    elif method == 'spearman':
        corr, p_value = spearmanr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'")

    # Interpret correlation strength
    abs_corr = abs(corr)
    if abs_corr < 0.3:
        strength = 'weak'
    elif abs_corr < 0.7:
        strength = 'moderate'
    else:
        strength = 'strong'

    direction = 'negative' if corr < 0 else 'positive'
    interpretation = f"{strength} {direction} correlation"

    return {
        'correlation': corr,
        'p_value': p_value,
        'method': method,
        'significant': p_value < 0.05,
        'interpretation': interpretation,
        'strength': strength,
        'direction': direction
    }


def test_significance(group1: List[float], group2: List[float],
                     test: str = 'ttest') -> Dict[str, float]:
    """
    Test statistical significance of difference between two groups.

    Parameters:
    -----------
    group1 : list of float
        First group of values (e.g., performance at low distance)
    group2 : list of float
        Second group of values (e.g., performance at high distance)
    test : str
        Statistical test: 'ttest' (parametric) or 'mannwhitney' (non-parametric)

    Returns:
    --------
    dict : Dictionary with test results
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    if len(group1) < 2 or len(group2) < 2:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'test': test,
            'significant': False,
            'interpretation': 'Insufficient samples'
        }

    if test == 'ttest':
        statistic, p_value = ttest_ind(group1, group2)
        test_name = "Independent t-test"
    elif test == 'mannwhitney':
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test: {test}. Use 'ttest' or 'mannwhitney'")

    mean_diff = np.mean(group1) - np.mean(group2)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'test': test_name,
        'significant': p_value < 0.05,
        'mean_difference': mean_diff,
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'interpretation': f"p={'<' if p_value < 0.05 else '≥'} 0.05, difference is {'significant' if p_value < 0.05 else 'not significant'}"
    }


def compute_confidence_intervals(data: List[float],
                                 confidence: float = 0.95,
                                 n_bootstrap: int = 1000) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for each bin.

    Parameters:
    -----------
    data : list of float
        Performance values for a bin
    confidence : float
        Confidence level (default: 0.95 for 95% CI)
    n_bootstrap : int
        Number of bootstrap samples

    Returns:
    --------
    dict : Dictionary with:
        - 'mean': Sample mean
        - 'lower': Lower bound of CI
        - 'upper': Upper bound of CI
        - 'std': Standard deviation
        - 'ci_width': Width of confidence interval
    """
    data = np.array(data)

    if len(data) < 2:
        return {
            'mean': np.mean(data) if len(data) > 0 else np.nan,
            'lower': np.nan,
            'upper': np.nan,
            'std': 0.0,
            'ci_width': 0.0,
            'n_samples': len(data)
        }

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = resample(data, replace=True, n_samples=len(data))
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        'mean': np.mean(data),
        'lower': lower,
        'upper': upper,
        'std': np.std(data),
        'ci_width': upper - lower,
        'n_samples': len(data),
        'confidence': confidence
    }


def detect_threshold(x: List[float], y: List[float],
                     performance_drop: float = 0.1,
                     method: str = 'absolute') -> Dict:
    """
    Identify distance threshold where performance drops significantly.

    Parameters:
    -----------
    x : list of float
        Distance values
    y : list of float
        Performance values
    performance_drop : float
        Threshold for significant drop (default: 0.1 = 10% drop)
    method : str
        'absolute' - absolute performance drop
        'relative' - relative drop from initial performance

    Returns:
    --------
    dict : Dictionary with threshold information
    """
    x = np.array(x)
    y = np.array(y)

    if len(y) < 2:
        return {
            'threshold_distance': np.nan,
            'threshold_index': -1,
            'initial_performance': np.nan,
            'threshold_performance': np.nan,
            'drop_amount': np.nan,
            'method': method
        }

    initial_perf = y[0]

    if method == 'absolute':
        threshold_perf = initial_perf - performance_drop
    elif method == 'relative':
        threshold_perf = initial_perf * (1 - performance_drop)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find first point where performance drops below threshold
    below_threshold = y < threshold_perf
    if np.any(below_threshold):
        threshold_idx = np.argmax(below_threshold)
        threshold_dist = x[threshold_idx]
        actual_perf = y[threshold_idx]
    else:
        # Performance never drops below threshold
        threshold_idx = len(y) - 1
        threshold_dist = x[-1]
        actual_perf = y[-1]

    return {
        'threshold_distance': threshold_dist,
        'threshold_index': threshold_idx,
        'initial_performance': initial_perf,
        'threshold_performance': actual_perf,
        'target_threshold': threshold_perf,
        'drop_amount': initial_perf - actual_perf,
        'relative_drop': (initial_perf - actual_perf) / initial_perf if initial_perf != 0 else 0,
        'method': method,
        'threshold_reached': actual_perf < threshold_perf
    }


def monotonicity_test(y: List[float]) -> Dict[str, float]:
    """
    Test if performance degradation is monotonic (consistently decreasing).

    Parameters:
    -----------
    y : list of float
        Performance values

    Returns:
    --------
    dict : Monotonicity metrics
    """
    y = np.array(y)

    if len(y) < 2:
        return {
            'is_monotonic': True,
            'monotonicity_score': 1.0,
            'violations': 0,
            'max_violation': 0.0
        }

    # Count violations (increases when should be decreasing)
    diffs = np.diff(y)
    violations = np.sum(diffs > 0)
    total_transitions = len(diffs)

    # Monotonicity score: fraction of non-increasing transitions
    monotonicity_score = 1.0 - (violations / total_transitions) if total_transitions > 0 else 1.0

    # Maximum violation magnitude
    max_violation = np.max(diffs[diffs > 0]) if np.any(diffs > 0) else 0.0

    return {
        'is_monotonic': violations == 0,
        'monotonicity_score': monotonicity_score,
        'violations': violations,
        'total_transitions': total_transitions,
        'max_violation': max_violation,
        'interpretation': f"{'Strictly monotonic' if violations == 0 else f'{violations} violations, score: {monotonicity_score:.2f}'}"
    }


def effect_size_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Parameters:
    -----------
    group1 : list of float
        First group (e.g., low distance performance)
    group2 : list of float
        Second group (e.g., high distance performance)

    Returns:
    --------
    float : Cohen's d effect size
        < 0.2: small effect
        0.2-0.8: medium effect
        > 0.8: large effect
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    if len(group1) < 2 or len(group2) < 2:
        return np.nan

    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                          (len(group2) - 1) * np.var(group2, ddof=1)) /
                         (len(group1) + len(group2) - 2))

    if pooled_std == 0:
        return np.nan

    cohens_d = mean_diff / pooled_std
    return cohens_d


def comprehensive_statistical_report(x: List[float], y: List[float],
                                     sample_counts: List[int] = None) -> Dict:
    """
    Generate comprehensive statistical analysis report.

    Parameters:
    -----------
    x : list of float
        Distance values
    y : list of float
        Performance values
    sample_counts : list of int, optional
        Number of samples per bin

    Returns:
    --------
    dict : Comprehensive statistical report
    """
    report = {}

    # Correlation analysis
    report['pearson'] = compute_correlation(x, y, method='pearson')
    report['spearman'] = compute_correlation(x, y, method='spearman')

    # Monotonicity
    report['monotonicity'] = monotonicity_test(y)

    # Compare first vs last bin (if enough data)
    if len(y) >= 2:
        # Use first 1/3 vs last 1/3 for comparison
        split_point = len(y) // 3
        group_low = y[:split_point] if split_point > 0 else [y[0]]
        group_high = y[-split_point:] if split_point > 0 else [y[-1]]

        report['group_comparison'] = test_significance(group_low, group_high, test='ttest')
        report['effect_size'] = {
            'cohens_d': effect_size_cohens_d(group_low, group_high),
            'interpretation': 'small (<0.2), medium (0.2-0.8), or large (>0.8)'
        }
    else:
        report['group_comparison'] = None
        report['effect_size'] = None

    # Threshold detection (10% drop)
    report['threshold_10pct'] = detect_threshold(x, y, performance_drop=0.1, method='relative')

    # Threshold detection (20% drop)
    report['threshold_20pct'] = detect_threshold(x, y, performance_drop=0.2, method='relative')

    # Summary statistics
    report['summary'] = {
        'mean_performance': np.mean(y),
        'std_performance': np.std(y),
        'min_performance': np.min(y),
        'max_performance': np.max(y),
        'performance_range': np.max(y) - np.min(y),
        'total_samples': np.sum(sample_counts) if sample_counts else len(y)
    }

    return report


def print_statistical_report(report: Dict):
    """
    Print formatted statistical report.

    Parameters:
    -----------
    report : dict
        Report from comprehensive_statistical_report()
    """
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS REPORT")
    print("=" * 60)

    # Correlation
    print("\nCORRELATION ANALYSIS:")
    print("-" * 60)
    pearson = report['pearson']
    print(f"Pearson correlation: r = {pearson['correlation']:.4f}, p = {pearson['p_value']:.4f}")
    print(f"  {pearson['interpretation']} {'(significant)' if pearson['significant'] else '(not significant)'}")

    spearman = report['spearman']
    print(f"Spearman correlation: ρ = {spearman['correlation']:.4f}, p = {spearman['p_value']:.4f}")
    print(f"  {spearman['interpretation']} {'(significant)' if spearman['significant'] else '(not significant)'}")

    # Monotonicity
    print("\nMONOTONICITY TEST:")
    print("-" * 60)
    mono = report['monotonicity']
    print(f"Monotonic degradation: {mono['is_monotonic']}")
    print(f"Monotonicity score: {mono['monotonicity_score']:.4f}")
    print(f"Violations: {mono['violations']} out of {mono['total_transitions']} transitions")

    # Group comparison
    if report['group_comparison']:
        print("\nGROUP COMPARISON (Low vs High Distance):")
        print("-" * 60)
        comp = report['group_comparison']
        print(f"Test: {comp['test']}")
        print(f"Low distance mean: {comp['group1_mean']:.4f}")
        print(f"High distance mean: {comp['group2_mean']:.4f}")
        print(f"Difference: {comp['mean_difference']:.4f}")
        print(f"p-value: {comp['p_value']:.4f} {'(significant)' if comp['significant'] else '(not significant)'}")

        if report['effect_size']:
            cohens_d = report['effect_size']['cohens_d']
            if not np.isnan(cohens_d):
                magnitude = 'small' if abs(cohens_d) < 0.2 else ('medium' if abs(cohens_d) < 0.8 else 'large')
                print(f"Effect size (Cohen's d): {cohens_d:.4f} ({magnitude})")

    # Thresholds
    print("\nPERFORMANCE DROP THRESHOLDS:")
    print("-" * 60)
    thresh_10 = report['threshold_10pct']
    print(f"10% drop threshold: distance ≤ {thresh_10['threshold_distance']:.4f}")
    print(f"  Initial: {thresh_10['initial_performance']:.4f} → Threshold: {thresh_10['threshold_performance']:.4f}")

    thresh_20 = report['threshold_20pct']
    print(f"20% drop threshold: distance ≤ {thresh_20['threshold_distance']:.4f}")
    print(f"  Initial: {thresh_20['initial_performance']:.4f} → Threshold: {thresh_20['threshold_performance']:.4f}")

    # Summary
    print("\nSUMMARY STATISTICS:")
    print("-" * 60)
    summary = report['summary']
    print(f"Mean performance: {summary['mean_performance']:.4f} ± {summary['std_performance']:.4f}")
    print(f"Range: [{summary['min_performance']:.4f}, {summary['max_performance']:.4f}]")
    print(f"Total samples: {summary['total_samples']}")

    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Example usage
    print("Statistical analysis module loaded successfully")

    # Test with dummy data
    x = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    y = np.array([0.95, 0.92, 0.88, 0.83, 0.78, 0.72, 0.65, 0.58])
    sample_counts = [150, 145, 148, 152, 149, 151, 147, 158]

    # Generate and print report
    report = comprehensive_statistical_report(x, y, sample_counts)
    print_statistical_report(report)

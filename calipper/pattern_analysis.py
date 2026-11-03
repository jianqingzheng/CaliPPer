'''
Pattern analysis module for generalization evaluation
Curve fitting and pattern characterization
Jianqing Zheng
2025.02.06
'''

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def fit_linear(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Fit linear model: y = a*x + b

    Parameters:
    -----------
    x : np.ndarray
        Independent variable (distance values)
    y : np.ndarray
        Dependent variable (performance values)

    Returns:
    --------
    dict : Dictionary with keys:
        - 'model': 'linear'
        - 'params': [slope, intercept]
        - 'r2': R² score
        - 'rmse': Root mean squared error
        - 'mae': Mean absolute error
        - 'predictions': Fitted values
    """
    x = np.array(x)
    y = np.array(y)

    # Fit linear model using numpy polyfit
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs

    # Predictions
    y_pred = slope * x + intercept

    # Compute metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    return {
        'model': 'linear',
        'params': {'slope': slope, 'intercept': intercept},
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred,
        'coefficients': coeffs
    }


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2) -> Dict:
    """
    Fit polynomial of specified degree.

    Parameters:
    -----------
    x : np.ndarray
        Independent variable (distance values)
    y : np.ndarray
        Dependent variable (performance values)
    degree : int
        Degree of polynomial (1-5 recommended)

    Returns:
    --------
    dict : Dictionary with fitting results and metrics
    """
    x = np.array(x)
    y = np.array(y)

    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)

    # Predictions
    y_pred = np.polyval(coeffs, x)

    # Compute metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    return {
        'model': f'polynomial_deg{degree}',
        'params': {'coefficients': coeffs, 'degree': degree},
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred,
        'coefficients': coeffs
    }


def fit_exponential_decay(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Fit exponential decay model: y = a * exp(-b * x) + c

    Tries multiple initial conditions to maximize convergence.

    Parameters:
    -----------
    x : np.ndarray
        Independent variable (distance values)
    y : np.ndarray
        Dependent variable (performance values)

    Returns:
    --------
    dict : Dictionary with fitting results and metrics
    """
    x = np.array(x)
    y = np.array(y)

    # Define exponential decay function
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    y_range = np.max(y) - np.min(y)
    x_range = np.ptp(x) if np.ptp(x) > 0 else 1.0

    # Try multiple initial conditions
    init_conditions = [
        (y_range, 1.0, np.min(y)),
        (y_range, 0.1, np.min(y)),
        (y_range, 10.0, np.min(y)),
        (y_range, 1.0 / x_range, np.min(y)),
        (y_range * 0.5, 0.5, np.mean(y)),
        (y_range, 2.0 / x_range, np.min(y)),
    ]

    best_result = None
    best_r2 = -np.inf

    for a_init, b_init, c_init in init_conditions:
        try:
            params, _ = curve_fit(
                exp_decay, x, y, p0=[a_init, b_init, c_init],
                maxfev=50000,
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))

            y_pred = exp_decay(x, *params)
            r2 = r2_score(y, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_result = {
                    'model': 'exponential_decay',
                    'params': {'a': params[0], 'b': params[1], 'c': params[2]},
                    'r2': r2,
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'mae': mean_absolute_error(y, y_pred),
                    'predictions': y_pred,
                    'success': True
                }
        except Exception:
            continue

    if best_result is not None:
        return best_result

    print(f"Exponential fit failed: all {len(init_conditions)} initial conditions failed")
    return {
        'model': 'exponential_decay',
        'params': None,
        'r2': -np.inf,
        'rmse': np.inf,
        'mae': np.inf,
        'predictions': None,
        'success': False
    }


def fit_sigmoid(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Fit sigmoid/logistic curve: y = L / (1 + exp(-k*(x-x0)))

    Parameters:
    -----------
    x : np.ndarray
        Independent variable (distance values)
    y : np.ndarray
        Dependent variable (performance values)

    Returns:
    --------
    dict : Dictionary with fitting results and metrics
    """
    x = np.array(x)
    y = np.array(y)

    # Define sigmoid function
    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Initial parameter guess
    L_init = np.max(y)
    k_init = 1.0
    x0_init = np.median(x)

    try:
        # Fit curve
        params, _ = curve_fit(sigmoid, x, y, p0=[L_init, k_init, x0_init], maxfev=10000)

        # Predictions
        y_pred = sigmoid(x, *params)

        # Compute metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        return {
            'model': 'sigmoid',
            'params': {'L': params[0], 'k': params[1], 'x0': params[2]},
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred,
            'success': True
        }
    except Exception as e:
        print(f"Sigmoid fit failed: {e}")
        return {
            'model': 'sigmoid',
            'params': None,
            'r2': -np.inf,
            'rmse': np.inf,
            'mae': np.inf,
            'predictions': None,
            'success': False
        }


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Fit isotonic (monotone decreasing) regression.

    Nonparametric, monotone by construction — ideal for degradation curves
    where performance is expected to decrease with distance.

    Uses linear interpolation between fitted points for prediction at new x.

    Parameters:
    -----------
    x : np.ndarray
        Independent variable (distance values)
    y : np.ndarray
        Dependent variable (performance values)

    Returns:
    --------
    dict : Dictionary with fitting results and metrics
    """
    from sklearn.isotonic import IsotonicRegression

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    try:
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
        y_pred = iso.fit_transform(x, y)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        return {
            'model': 'isotonic',
            'params': {'x_knots': iso.X_thresholds_.tolist(),
                       'y_knots': iso.y_thresholds_.tolist()},
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred,
            'success': True,
        }
    except Exception as e:
        print(f"Isotonic fit failed: {e}")
        return {
            'model': 'isotonic',
            'params': None,
            'r2': -np.inf,
            'rmse': np.inf,
            'mae': np.inf,
            'predictions': None,
            'success': False,
        }


def select_best_fit(x: List[float], y: List[float],
                    models: List[str] = ['linear', 'exponential', 'isotonic']) -> Tuple[str, Dict]:
    """
    Try multiple models and select best by R².

    Parameters:
    -----------
    x : list of float
        Independent variable (distance values)
    y : list of float
        Dependent variable (performance values)
    models : list of str
        Models to try: 'linear', 'poly2', 'poly3', 'poly4', 'poly5',
        'exponential', 'sigmoid', 'isotonic'

    Returns:
    --------
    tuple : (best_model_name, best_fit_results)
    """
    x = np.array(x)
    y = np.array(y)

    results = {}

    for model_name in models:
        if model_name == 'linear':
            results[model_name] = fit_linear(x, y)
        elif model_name.startswith('poly'):
            degree = int(model_name.replace('poly', ''))
            results[model_name] = fit_polynomial(x, y, degree=degree)
        elif model_name == 'exponential':
            results[model_name] = fit_exponential_decay(x, y)
        elif model_name == 'sigmoid':
            results[model_name] = fit_sigmoid(x, y)
        elif model_name == 'isotonic':
            results[model_name] = fit_isotonic(x, y)
        else:
            print(f"Unknown model: {model_name}")

    # Select best model by R²
    best_model = None
    best_r2 = -np.inf

    for model_name, result in results.items():
        if result['r2'] >= best_r2:
            best_r2 = result['r2']
            best_model = model_name

    if best_model is None:
        # All fits failed — return first result with failure info
        first_key = next(iter(results))
        return first_key, results[first_key]

    return best_model, results[best_model]


def fit_prefer_exponential(x: List[float], y: List[float]) -> Tuple[str, Dict]:
    """
    Try exponential decay first; fall back to isotonic only if exponential
    fails to converge.  This ensures we use a parametric model whenever
    possible and only resort to isotonic as a last resort.
    """
    name, result = select_best_fit(x, y, ['exponential'])
    if result.get('success', False):
        return name, result
    # Exponential failed — fall back to isotonic
    return select_best_fit(x, y, ['isotonic'])


def classify_pattern(x: List[float], y: List[float]) -> Tuple[str, float]:
    """
    Classify degradation pattern as:
    - 'linear' - steady decline
    - 'exponential' - rapid initial decline
    - 'plateau' - maintains performance
    - 'step' - sudden drop

    Parameters:
    -----------
    x : list of float
        Distance values
    y : list of float
        Performance values

    Returns:
    --------
    tuple : (pattern_type, confidence)
        pattern_type: str describing the pattern
        confidence: float in [0, 1] indicating classification confidence
    """
    x = np.array(x)
    y = np.array(y)

    # Fit multiple models
    _, best_fit = select_best_fit(x, y, models=['linear', 'exponential', 'isotonic'])

    # Calculate degradation metrics
    total_change = y[0] - y[-1] if len(y) > 0 else 0
    relative_change = total_change / y[0] if y[0] != 0 else 0

    # Compute rate of change
    dy = np.diff(y)
    dx = np.diff(x)
    slopes = dy / (dx + 1e-10)
    slope_variance = np.var(slopes) if len(slopes) > 0 else 0

    # Classification logic
    if abs(relative_change) < 0.05:
        pattern = 'plateau'
        confidence = 1.0 - abs(relative_change) / 0.05
    elif best_fit['model'] == 'exponential' and best_fit['r2'] > 0.85:
        pattern = 'exponential'
        confidence = best_fit['r2']
    elif best_fit['model'] == 'linear' and best_fit['r2'] > 0.85:
        pattern = 'linear'
        confidence = best_fit['r2']
    elif slope_variance > np.mean(np.abs(slopes)) ** 2:
        pattern = 'step'
        confidence = min(1.0, slope_variance / (np.mean(np.abs(slopes)) ** 2))
    else:
        pattern = 'mixed'
        confidence = best_fit['r2']

    return pattern, confidence


def find_inflection_points(x: List[float], y: List[float]) -> List[float]:
    """
    Find where degradation rate changes significantly (inflection points).

    Parameters:
    -----------
    x : list of float
        Distance values
    y : list of float
        Performance values

    Returns:
    --------
    list of float : x values where inflection occurs
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 4:
        return []

    # Compute second derivative (curvature)
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)

    # Find local extrema of second derivative
    # These correspond to inflection points
    inflection_indices = argrelextrema(np.abs(d2y), np.greater)[0]

    # Filter significant inflection points
    threshold = np.std(d2y) * 0.5
    significant_inflections = [idx for idx in inflection_indices if np.abs(d2y[idx]) > threshold]

    return [x[idx] for idx in significant_inflections]


def compute_degradation_rate(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Compute slope of decline.

    Parameters:
    -----------
    x : list of float
        Distance values
    y : list of float
        Performance values

    Returns:
    --------
    dict : Dictionary with:
        - 'mean_rate': Average degradation rate
        - 'rate_variance': Variance in degradation rate
        - 'total_change': Total performance change
        - 'relative_change': Total change relative to initial performance
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 2:
        return {
            'mean_rate': 0.0,
            'rate_variance': 0.0,
            'total_change': 0.0,
            'relative_change': 0.0
        }

    # Compute rates between consecutive points
    dy = np.diff(y)
    dx = np.diff(x)
    rates = dy / (dx + 1e-10)

    mean_rate = np.mean(rates)
    rate_variance = np.var(rates)
    total_change = y[-1] - y[0]
    relative_change = total_change / y[0] if y[0] != 0 else 0.0

    return {
        'mean_rate': mean_rate,
        'rate_variance': rate_variance,
        'total_change': total_change,
        'relative_change': relative_change
    }


def identify_reliable_region(x: List[float], y: List[float], threshold: float = 0.8) -> Tuple[float, Tuple[float, float]]:
    """
    Find distance range where performance > threshold * max_performance.

    Parameters:
    -----------
    x : list of float
        Distance values
    y : list of float
        Performance values
    threshold : float
        Fraction of maximum performance (0-1)

    Returns:
    --------
    tuple : (max_reliable_distance, (min_distance, max_distance))
        max_reliable_distance: Maximum distance maintaining threshold performance
        (min_distance, max_distance): Range of reliable distances
    """
    x = np.array(x)
    y = np.array(y)

    if len(y) == 0:
        return 0.0, (0.0, 0.0)

    max_perf = np.max(y)
    threshold_value = threshold * max_perf

    # Find all points above threshold
    reliable_indices = np.where(y >= threshold_value)[0]

    if len(reliable_indices) == 0:
        return x[0], (x[0], x[0])

    # Get min and max distance in reliable region
    min_reliable_dist = x[reliable_indices[0]]
    max_reliable_dist = x[reliable_indices[-1]]

    return max_reliable_dist, (min_reliable_dist, max_reliable_dist)


def compute_goodness_of_fit(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive goodness-of-fit metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    dict : Dictionary with metrics (R², RMSE, MAE, etc.)
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Additional metrics
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Adjusted R²
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else r2

    return {
        'r2': r2,
        'adj_r2': adj_r2,
        'rmse': rmse,
        'mae': mae,
        'ss_res': ss_res,
        'ss_tot': ss_tot
    }


if __name__ == '__main__':
    # Example usage
    print("Pattern analysis module loaded successfully")

    # Test with dummy data
    x = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    y = np.array([0.95, 0.92, 0.88, 0.83, 0.78, 0.72, 0.65, 0.58])

    # Test pattern classification
    pattern, confidence = classify_pattern(x, y)
    print(f"Pattern: {pattern}, Confidence: {confidence:.2f}")

    # Test best fit selection
    best_model, best_fit = select_best_fit(x, y)
    print(f"Best model: {best_model}, R²: {best_fit['r2']:.3f}")

    # Test degradation rate
    deg_rate = compute_degradation_rate(x, y)
    print(f"Mean degradation rate: {deg_rate['mean_rate']:.3f}")

    # Test reliable region
    max_dist, (min_dist, max_dist_range) = identify_reliable_region(x, y, threshold=0.8)
    print(f"Reliable region: up to distance {max_dist:.3f}")

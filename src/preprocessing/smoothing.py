"""
Savitzky-Golay smoothing utilities for NDVI time series.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Union, Optional


def smooth_series(series: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to a time series.
    
    Args:
        series: Input time series data
        window: Window size for smoothing (must be odd and > polyorder)
        polyorder: Polynomial order for fitting
        
    Returns:
        Smoothed time series
    """
    if len(series) < window:
        # If series is too short, return original
        return series
    
    # Ensure window is odd and greater than polyorder
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        window = polyorder + 2
    
    try:
        smoothed = savgol_filter(series, window, polyorder)
        return smoothed
    except Exception as e:
        print(f"Warning: Smoothing failed ({e}), returning original series")
        return series


def smooth_dataframe(df, value_col: str, group_col: str, 
                    window: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Apply smoothing to grouped data in a DataFrame.
    
    Args:
        df: Input DataFrame
        value_col: Column name containing values to smooth
        group_col: Column name to group by (e.g., 'location_id')
        window: Window size for smoothing
        polyorder: Polynomial order for fitting
        
    Returns:
        Array of smoothed values
    """
    smoothed_values = []
    
    for group_id, group_df in df.groupby(group_col):
        group_values = group_df[value_col].values
        
        # Remove NaN values for smoothing
        valid_mask = ~np.isnan(group_values)
        if valid_mask.sum() < window:
            # Not enough valid points for smoothing
            smoothed_values.extend(group_values)
            continue
        
        valid_values = group_values[valid_mask]
        smoothed_valid = smooth_series(valid_values, window, polyorder)
        
        # Reconstruct full series with NaN handling
        full_smoothed = np.full_like(group_values, np.nan)
        full_smoothed[valid_mask] = smoothed_valid
        smoothed_values.extend(full_smoothed)
    
    return np.array(smoothed_values)


def detect_outliers(series: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers in a time series using Z-score method.
    
    Args:
        series: Input time series
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if len(series) < 3:
        return np.zeros(len(series), dtype=bool)
    
    # Calculate Z-scores
    mean_val = np.nanmean(series)
    std_val = np.nanstd(series)
    
    if std_val == 0:
        return np.zeros(len(series), dtype=bool)
    
    z_scores = np.abs((series - mean_val) / std_val)
    return z_scores > threshold


def interpolate_missing(series: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Interpolate missing values in a time series.
    
    Args:
        series: Input time series with potential NaN values
        method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        Series with interpolated values
    """
    if not np.any(np.isnan(series)):
        return series
    
    valid_indices = ~np.isnan(series)
    valid_values = series[valid_indices]
    
    if len(valid_values) < 2:
        # Not enough valid points for interpolation
        return series
    
    # Create interpolation function
    from scipy.interpolate import interp1d
    
    try:
        if method == 'linear':
            f = interp1d(np.where(valid_indices)[0], valid_values, 
                        kind='linear', bounds_error=False, fill_value='extrapolate')
        elif method == 'cubic':
            f = interp1d(np.where(valid_indices)[0], valid_values, 
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
        else:  # nearest
            f = interp1d(np.where(valid_indices)[0], valid_values, 
                        kind='nearest', bounds_error=False, fill_value='extrapolate')
        
        # Interpolate all values
        interpolated = f(np.arange(len(series)))
        
        # Only replace NaN values
        result = series.copy()
        result[np.isnan(series)] = interpolated[np.isnan(series)]
        
        return result
        
    except Exception as e:
        print(f"Warning: Interpolation failed ({e}), returning original series")
        return series

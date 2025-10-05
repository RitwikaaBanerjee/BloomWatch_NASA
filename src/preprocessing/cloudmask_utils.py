"""
Cloud masking utilities for satellite data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def apply_cloud_mask(ndvi_values: np.ndarray, 
                    quality_flags: Optional[np.ndarray] = None,
                    cloud_threshold: float = 0.1) -> np.ndarray:
    """
    Apply cloud masking to NDVI values.
    
    Args:
        ndvi_values: NDVI values
        quality_flags: Quality flags (if available)
        cloud_threshold: Threshold for cloud detection (NDVI < threshold = cloud)
        
    Returns:
        Masked NDVI values (NaN for cloudy pixels)
    """
    masked_ndvi = ndvi_values.copy()
    
    # Basic cloud masking based on NDVI threshold
    cloud_mask = ndvi_values < cloud_threshold
    masked_ndvi[cloud_mask] = np.nan
    
    # Additional quality flag masking if available
    if quality_flags is not None:
        # Assume higher quality flag values indicate better quality
        # This is product-specific and may need adjustment
        quality_mask = quality_flags < 2  # Adjust threshold as needed
        masked_ndvi[quality_mask] = np.nan
    
    return masked_ndvi


def detect_clouds_modis(ndvi: np.ndarray, 
                       reliability: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Detect clouds in MODIS data using NDVI and reliability flags.
    
    Args:
        ndvi: NDVI values
        reliability: MODIS reliability flags (0-3, where 3 is best)
        
    Returns:
        Boolean array indicating cloudy pixels
    """
    cloud_mask = np.zeros_like(ndvi, dtype=bool)
    
    # Low NDVI values often indicate clouds
    cloud_mask |= ndvi < 0.1
    
    # Use reliability flags if available
    if reliability is not None:
        # MODIS reliability: 0=no data, 1=low, 2=medium, 3=high
        cloud_mask |= reliability < 2
    
    return cloud_mask


def temporal_consistency_filter(ndvi_series: np.ndarray, 
                               window: int = 3,
                               threshold: float = 0.3) -> np.ndarray:
    """
    Filter out temporally inconsistent NDVI values.
    
    Args:
        ndvi_series: Time series of NDVI values
        window: Window size for consistency check
        threshold: Maximum allowed change between consecutive values
        
    Returns:
        Filtered NDVI series with outliers set to NaN
    """
    filtered_series = ndvi_series.copy()
    
    for i in range(1, len(ndvi_series)):
        if np.isnan(ndvi_series[i]) or np.isnan(ndvi_series[i-1]):
            continue
        
        # Check for large jumps
        change = abs(ndvi_series[i] - ndvi_series[i-1])
        if change > threshold:
            # Check surrounding values for context
            start_idx = max(0, i - window // 2)
            end_idx = min(len(ndvi_series), i + window // 2 + 1)
            
            surrounding_values = ndvi_series[start_idx:end_idx]
            valid_values = surrounding_values[~np.isnan(surrounding_values)]
            
            if len(valid_values) > 1:
                mean_val = np.mean(valid_values)
                if abs(ndvi_series[i] - mean_val) > threshold:
                    filtered_series[i] = np.nan
    
    return filtered_series


def seasonal_outlier_detection(ndvi_series: np.ndarray, 
                              dates: pd.DatetimeIndex,
                              threshold: float = 2.0) -> np.ndarray:
    """
    Detect outliers based on seasonal patterns.
    
    Args:
        ndvi_series: Time series of NDVI values
        dates: Corresponding dates
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if len(ndvi_series) < 12:  # Need at least a year of data
        return np.zeros(len(ndvi_series), dtype=bool)
    
    outliers = np.zeros(len(ndvi_series), dtype=bool)
    
    # Group by month
    monthly_groups = {}
    for i, date in enumerate(dates):
        month = date.month
        if month not in monthly_groups:
            monthly_groups[month] = []
        monthly_groups[month].append(i)
    
    # Check each month for outliers
    for month, indices in monthly_groups.items():
        if len(indices) < 3:  # Need at least 3 samples
            continue
        
        month_values = ndvi_series[indices]
        valid_values = month_values[~np.isnan(month_values)]
        
        if len(valid_values) < 3:
            continue
        
        # Calculate Z-scores
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        
        if std_val == 0:
            continue
        
        z_scores = np.abs((month_values - mean_val) / std_val)
        month_outliers = z_scores > threshold
        
        # Mark outliers
        for j, is_outlier in enumerate(month_outliers):
            if is_outlier:
                outliers[indices[j]] = True
    
    return outliers

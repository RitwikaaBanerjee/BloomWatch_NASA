"""
Feature engineering for BloomWatch ML models.
Creates temporal, seasonal, and statistical features from NDVI time series.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_PROCESSED_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date column.
    
    Args:
        df: DataFrame with date column
        
    Returns:
        DataFrame with additional temporal features
    """
    df_features = df.copy()
    
    # Extract date components
    df_features['year'] = df_features['date'].dt.year
    df_features['month'] = df_features['date'].dt.month
    df_features['day_of_year'] = df_features['date'].dt.dayofyear
    df_features['week_of_year'] = df_features['date'].dt.isocalendar().week
    
    # Cyclical encoding for seasonal patterns
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['doy_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365.25)
    df_features['doy_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365.25)
    
    # Season indicators
    df_features['is_spring'] = df_features['month'].isin([3, 4, 5]).astype(int)
    df_features['is_summer'] = df_features['month'].isin([6, 7, 8]).astype(int)
    df_features['is_autumn'] = df_features['month'].isin([9, 10, 11]).astype(int)
    df_features['is_winter'] = df_features['month'].isin([12, 1, 2]).astype(int)
    
    return df_features


def create_rolling_features(df: pd.DataFrame, 
                          value_col: str = 'ndvi_smoothed',
                          windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Create rolling statistical features.
    
    Args:
        df: DataFrame with NDVI data
        value_col: Column name containing values
        windows: List of window sizes for rolling calculations
        
    Returns:
        DataFrame with rolling features
    """
    df_features = df.copy()
    
    for window in windows:
        # Rolling mean
        df_features[f'ndvi_mean_{window}'] = (
            df_features.groupby('location_id')[value_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Rolling standard deviation
        df_features[f'ndvi_std_{window}'] = (
            df_features.groupby('location_id')[value_col]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        # Rolling minimum
        df_features[f'ndvi_min_{window}'] = (
            df_features.groupby('location_id')[value_col]
            .rolling(window=window, min_periods=1)
            .min()
            .reset_index(0, drop=True)
        )
        
        # Rolling maximum
        df_features[f'ndvi_max_{window}'] = (
            df_features.groupby('location_id')[value_col]
            .rolling(window=window, min_periods=1)
            .max()
            .reset_index(0, drop=True)
        )
        
        # Rolling slope (linear trend)
        def calculate_slope(series):
            if len(series) < 2:
                return np.nan
            x = np.arange(len(series))
            valid_mask = ~np.isnan(series)
            if valid_mask.sum() < 2:
                return np.nan
            x_valid = x[valid_mask]
            y_valid = series[valid_mask]
            return np.polyfit(x_valid, y_valid, 1)[0]
        
        df_features[f'ndvi_slope_{window}'] = (
            df_features.groupby('location_id')[value_col]
            .rolling(window=window, min_periods=2)
            .apply(calculate_slope, raw=False)
            .reset_index(0, drop=True)
        )
    
    return df_features


def create_lag_features(df: pd.DataFrame, 
                       value_col: str = 'ndvi_smoothed',
                       lags: List[int] = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    """
    Create lag features (previous values).
    
    Args:
        df: DataFrame with NDVI data
        value_col: Column name containing values
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    df_features = df.copy()
    
    for lag in lags:
        df_features[f'ndvi_lag_{lag}'] = (
            df_features.groupby('location_id')[value_col]
            .shift(lag)
        )
    
    return df_features


def create_difference_features(df: pd.DataFrame, 
                             value_col: str = 'ndvi_smoothed',
                             periods: List[int] = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    """
    Create difference features (change from previous values).
    
    Args:
        df: DataFrame with NDVI data
        value_col: Column name containing values
        periods: List of difference periods
        
    Returns:
        DataFrame with difference features
    """
    df_features = df.copy()
    
    for period in periods:
        df_features[f'ndvi_diff_{period}'] = (
            df_features.groupby('location_id')[value_col]
            .diff(period)
        )
    
    return df_features


def create_peak_features(df: pd.DataFrame, 
                        value_col: str = 'ndvi_smoothed') -> pd.DataFrame:
    """
    Create features related to NDVI peaks and valleys.
    
    Args:
        df: DataFrame with NDVI data
        value_col: Column name containing values
        
    Returns:
        DataFrame with peak features
    """
    df_features = df.copy()
    
    # Calculate rolling peak detection
    def find_peaks(series, window=5):
        if len(series) < window:
            return np.zeros(len(series), dtype=bool)
        
        peaks = np.zeros(len(series), dtype=bool)
        for i in range(window//2, len(series) - window//2):
            center_val = series.iloc[i]
            window_vals = series.iloc[i-window//2:i+window//2+1]
            if center_val == window_vals.max() and center_val > 0.5:  # High NDVI threshold
                peaks[i] = True
        
        return peaks
    
    # Peak detection
    df_features['is_peak'] = (
        df_features.groupby('location_id')[value_col]
        .rolling(window=5, min_periods=3)
        .apply(lambda x: find_peaks(x).any(), raw=False)
        .reset_index(0, drop=True)
    )
    
    # Distance to last peak
    def distance_to_last_peak(series):
        peaks = find_peaks(series)
        if not peaks.any():
            return np.nan
        
        last_peak_idx = np.where(peaks)[0][-1]
        return len(series) - 1 - last_peak_idx
    
    df_features['days_since_peak'] = (
        df_features.groupby('location_id')[value_col]
        .rolling(window=12, min_periods=6)
        .apply(distance_to_last_peak, raw=False)
        .reset_index(0, drop=True)
    )
    
    return df_features


def create_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create climate-related features (placeholder for future climate data integration).
    
    Args:
        df: DataFrame with location data
        
    Returns:
        DataFrame with climate features
    """
    df_features = df.copy()
    
    # Simple latitude-based climate features
    df_features['latitude_abs'] = np.abs(df_features['latitude'])
    df_features['is_tropical'] = (df_features['latitude_abs'] < 23.5).astype(int)
    df_features['is_temperate'] = ((df_features['latitude_abs'] >= 23.5) & 
                                  (df_features['latitude_abs'] < 66.5)).astype(int)
    df_features['is_polar'] = (df_features['latitude_abs'] >= 66.5).astype(int)
    
    # Elevation features (placeholder - would need elevation data)
    df_features['elevation'] = 0  # Placeholder
    
    return df_features


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different variables.
    
    Args:
        df: DataFrame with existing features
        
    Returns:
        DataFrame with interaction features
    """
    df_features = df.copy()
    
    # NDVI-season interactions
    if 'ndvi_smoothed' in df_features.columns and 'month' in df_features.columns:
        df_features['ndvi_month_interaction'] = (
            df_features['ndvi_smoothed'] * df_features['month']
        )
    
    # NDVI-latitude interactions
    if 'ndvi_smoothed' in df_features.columns and 'latitude' in df_features.columns:
        df_features['ndvi_lat_interaction'] = (
            df_features['ndvi_smoothed'] * df_features['latitude']
        )
    
    # Rolling mean and slope interaction
    if 'ndvi_mean_3' in df_features.columns and 'ndvi_slope_3' in df_features.columns:
        df_features['mean_slope_interaction'] = (
            df_features['ndvi_mean_3'] * df_features['ndvi_slope_3']
        )
    
    return df_features


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features for the dataset.
    
    Args:
        df: DataFrame with processed NDVI data
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Creating temporal features...")
    df_features = create_temporal_features(df)
    
    logger.info("Creating rolling features...")
    df_features = create_rolling_features(df_features)
    
    logger.info("Creating lag features...")
    df_features = create_lag_features(df_features)
    
    logger.info("Creating difference features...")
    df_features = create_difference_features(df_features)
    
    logger.info("Creating peak features...")
    df_features = create_peak_features(df_features)
    
    logger.info("Creating climate features...")
    df_features = create_climate_features(df_features)
    
    logger.info("Creating interaction features...")
    df_features = create_interaction_features(df_features)
    
    # Remove rows with too many NaN values
    feature_cols = [col for col in df_features.columns 
                   if col not in ['location_id', 'latitude', 'longitude', 'date', 'ndvi_raw', 'ndvi_smoothed']]
    
    # Count NaN values per row
    nan_counts = df_features[feature_cols].isnull().sum(axis=1)
    
    # Keep rows with less than 50% NaN values
    valid_mask = nan_counts < len(feature_cols) * 0.5
    df_features = df_features[valid_mask].reset_index(drop=True)
    
    logger.info(f"Removed {sum(~valid_mask)} rows with too many missing features")
    logger.info(f"Final dataset: {len(df_features)} records with {len(feature_cols)} features")
    
    return df_features


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save features to CSV file."""
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        
        # Print feature summary
        feature_cols = [col for col in df.columns 
                       if col not in ['location_id', 'latitude', 'longitude', 'date', 'ndvi_raw', 'ndvi_smoothed']]
        
        logger.info(f"Created {len(feature_cols)} features:")
        for col in sorted(feature_cols):
            non_null_count = df[col].notna().sum()
            logger.info(f"  {col}: {non_null_count}/{len(df)} non-null values")
        
    except Exception as e:
        logger.error(f"Failed to save features: {e}")
        sys.exit(1)


def main():
    """Main feature engineering function."""
    parser = argparse.ArgumentParser(description='Create ML features from processed NDVI data')
    parser.add_argument('--input', required=True, 
                       help='Input processed CSV file path')
    parser.add_argument('--output', 
                       help='Output features CSV file path (auto-generated if not provided)')
    parser.add_argument('--rolling-windows', nargs='+', type=int, default=[3, 6, 12],
                       help='Rolling window sizes (default: 3 6 12)')
    parser.add_argument('--lags', nargs='+', type=int, default=[1, 2, 3, 6, 12],
                       help='Lag periods (default: 1 2 3 6 12)')
    parser.add_argument('--diff-periods', nargs='+', type=int, default=[1, 2, 3, 6, 12],
                       help='Difference periods (default: 1 2 3 6 12)')
    
    args = parser.parse_args()
    
    # Set up paths
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DATA_PROCESSED_DIR / f"{input_path.stem}_features.csv"
    
    # Load processed data
    logger.info(f"Loading processed data from {input_path}")
    try:
        df = pd.read_csv(input_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Validate data
    required_cols = ['location_id', 'date', 'ndvi_smoothed', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Create features
    df_features = create_all_features(df)
    
    # Save features
    save_features(df_features, output_path)
    
    logger.info("Feature engineering completed!")


if __name__ == '__main__':
    main()

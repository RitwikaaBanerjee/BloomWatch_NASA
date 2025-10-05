"""
NDVI preprocessing pipeline for BloomWatch.
Handles data cleaning, smoothing, and resampling.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    DATA_PROCESSED_DIR,
    NDVI_SCALE_FACTOR,
    DEFAULT_FREQUENCY,
    SMOOTHING_WINDOW,
    SMOOTHING_POLYORDER
)
from src.preprocessing.smoothing import smooth_dataframe, interpolate_missing
from src.preprocessing.cloudmask_utils import (
    temporal_consistency_filter,
    seasonal_outlier_detection
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Load raw NDVI data from CSV file."""
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {input_path}: {e}")
        sys.exit(1)


def validate_data(df: pd.DataFrame) -> bool:
    """Validate that required columns exist in the data."""
    required_cols = ['latitude', 'longitude', 'date', 'ndvi_raw']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data."""
    logger.info("Cleaning data...")
    
    # Create a copy
    df_clean = df.copy()
    
    # Convert date to datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Sort by location and date
    df_clean = df_clean.sort_values(['latitude', 'longitude', 'date']).reset_index(drop=True)
    
    # Create location ID
    df_clean['location_id'] = (
        df_clean['latitude'].round(4).astype(str) + '_' + 
        df_clean['longitude'].round(4).astype(str)
    )
    
    # Apply NDVI scaling if not already applied
    if df_clean['ndvi_raw'].max() > 1.0:
        logger.info("Applying NDVI scaling factor")
        df_clean['ndvi_raw'] = df_clean['ndvi_raw'] * NDVI_SCALE_FACTOR
    
    # Clip NDVI values to valid range
    df_clean['ndvi_raw'] = df_clean['ndvi_raw'].clip(-1.0, 1.0)
    
    # Remove invalid values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['latitude', 'longitude', 'date', 'ndvi_raw'])
    removed_count = initial_count - len(df_clean)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} records with missing values")
    
    return df_clean


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to remove outliers and inconsistent data."""
    logger.info("Applying quality filters...")
    
    df_filtered = df.copy()
    
    # Group by location for processing
    for location_id, group_df in tqdm(df_filtered.groupby('location_id'), 
                                     desc="Filtering locations"):
        if len(group_df) < 3:  # Skip locations with too few points
            continue
        
        # Sort by date
        group_df = group_df.sort_values('date')
        
        # Apply temporal consistency filter
        ndvi_values = group_df['ndvi_raw'].values
        filtered_values = temporal_consistency_filter(ndvi_values)
        
        # Update the dataframe
        mask = df_filtered['location_id'] == location_id
        df_filtered.loc[mask, 'ndvi_raw'] = filtered_values
    
    # Remove records that became NaN after filtering
    initial_count = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=['ndvi_raw'])
    removed_count = initial_count - len(df_filtered)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} records after quality filtering")
    
    return df_filtered


def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing values for each location."""
    logger.info("Interpolating missing values...")
    
    df_interp = df.copy()
    
    for location_id, group_df in tqdm(df_interp.groupby('location_id'), 
                                     desc="Interpolating locations"):
        if len(group_df) < 2:  # Skip locations with too few points
            continue
        
        # Sort by date
        group_df = group_df.sort_values('date')
        
        # Interpolate missing values
        ndvi_values = group_df['ndvi_raw'].values
        interpolated_values = interpolate_missing(ndvi_values, method='linear')
        
        # Update the dataframe
        mask = df_interp['location_id'] == location_id
        df_interp.loc[mask, 'ndvi_raw'] = interpolated_values
    
    return df_interp


def resample_to_frequency(df: pd.DataFrame, frequency: str = 'M') -> pd.DataFrame:
    """Resample data to specified frequency."""
    logger.info(f"Resampling to {frequency} frequency...")
    
    resampled_data = []
    
    for location_id, group_df in tqdm(df.groupby('location_id'), 
                                     desc="Resampling locations"):
        if len(group_df) < 2:
            continue
        
        # Sort by date and ensure datetime index
        group_df = group_df.sort_values('date').copy()
        group_df['date'] = pd.to_datetime(group_df['date'])
        group_df = group_df.set_index('date')
        
        # Resample to monthly frequency - use 'MS' for month start (more compatible)
        try:
            resampled = group_df['ndvi_raw'].resample('MS').mean()
        except:
            # Fallback if resampling fails
            resampled = group_df['ndvi_raw']
        
        # Create new dataframe
        resampled_df = pd.DataFrame({
            'location_id': location_id,
            'latitude': group_df['latitude'].iloc[0],
            'longitude': group_df['longitude'].iloc[0],
            'date': resampled.index,
            'ndvi_raw': resampled.values
        }).reset_index(drop=True)
        
        resampled_data.append(resampled_df)
    
    if resampled_data:
        df_resampled = pd.concat(resampled_data, ignore_index=True)
        logger.info(f"Resampled to {len(df_resampled)} records")
        return df_resampled
    else:
        logger.warning("No data after resampling")
        return df


def apply_smoothing(df: pd.DataFrame, window: int = 5, polyorder: int = 2) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to NDVI time series."""
    logger.info(f"Applying smoothing (window={window}, polyorder={polyorder})...")
    
    df_smooth = df.copy()
    
    # Apply smoothing to each location
    smoothed_values = smooth_dataframe(
        df_smooth, 
        value_col='ndvi_raw', 
        group_col='location_id',
        window=window,
        polyorder=polyorder
    )
    
    df_smooth['ndvi_smoothed'] = smoothed_values
    
    # Fill any remaining NaN values with original values
    df_smooth['ndvi_smoothed'] = df_smooth['ndvi_smoothed'].fillna(df_smooth['ndvi_raw'])
    
    return df_smooth


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save processed data to CSV file."""
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        # Print summary statistics
        logger.info(f"Final dataset: {len(df)} records")
        logger.info(f"Locations: {df['location_id'].nunique()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"NDVI range: {df['ndvi_smoothed'].min():.3f} to {df['ndvi_smoothed'].max():.3f}")
        
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        sys.exit(1)


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess NDVI data')
    parser.add_argument('--input', required=True, 
                       help='Input CSV file path')
    parser.add_argument('--output', 
                       help='Output CSV file path (auto-generated if not provided)')
    parser.add_argument('--frequency', default=DEFAULT_FREQUENCY,
                       help=f'Resampling frequency (default: {DEFAULT_FREQUENCY})')
    parser.add_argument('--smoothing-window', type=int, default=SMOOTHING_WINDOW,
                       help=f'Smoothing window size (default: {SMOOTHING_WINDOW})')
    parser.add_argument('--smoothing-polyorder', type=int, default=SMOOTHING_POLYORDER,
                       help=f'Smoothing polynomial order (default: {SMOOTHING_POLYORDER})')
    parser.add_argument('--skip-quality-filters', action='store_true',
                       help='Skip quality filtering step')
    
    args = parser.parse_args()
    
    # Set up paths
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DATA_PROCESSED_DIR / f"{input_path.stem}_processed.csv"
    
    # Load and validate data
    logger.info("Loading raw data...")
    df = load_raw_data(input_path)
    
    if not validate_data(df):
        sys.exit(1)
    
    # Clean data
    df = clean_data(df)
    
    # Apply quality filters (optional)
    if not args.skip_quality_filters:
        df = apply_quality_filters(df)
    
    # Interpolate missing values
    df = interpolate_missing_values(df)
    
    # Resample to specified frequency
    df = resample_to_frequency(df, args.frequency)
    
    # Apply smoothing
    df = apply_smoothing(df, args.smoothing_window, args.smoothing_polyorder)
    
    # Save processed data
    save_processed_data(df, output_path)
    
    logger.info("Preprocessing completed successfully!")


if __name__ == '__main__':
    main()

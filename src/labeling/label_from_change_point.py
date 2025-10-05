"""
Change-point detection for bloom onset labeling.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from ruptures import Pelt, Binseg, Window
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_PROCESSED_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_change_points(ndvi_series: np.ndarray, 
                        method: str = 'pelt',
                        min_size: int = 2,
                        penalty: float = 10.0) -> List[int]:
    """
    Detect change points in NDVI time series.
    
    Args:
        ndvi_series: NDVI time series
        method: Change point detection method ('pelt', 'binseg', 'window')
        min_size: Minimum segment size
        penalty: Penalty parameter for change point detection
        
    Returns:
        List of change point indices
    """
    if len(ndvi_series) < 6:  # Need at least 6 points for reliable detection
        return []
    
    # Remove NaN values for change point detection
    valid_mask = ~np.isnan(ndvi_series)
    if valid_mask.sum() < 6:
        return []
    
    valid_ndvi = ndvi_series[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    try:
        if method == 'pelt':
            model = Pelt(model="rbf", min_size=min_size)
            change_points = model.fit_predict(valid_ndvi, pen=penalty)
        elif method == 'binseg':
            model = Binseg(model="rbf", min_size=min_size)
            change_points = model.fit_predict(valid_ndvi, pen=penalty)
        elif method == 'window':
            model = Window(model="rbf", min_size=min_size)
            change_points = model.fit_predict(valid_ndvi, width=min(len(valid_ndvi)//2, 10))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to original indices
        if len(change_points) > 0 and change_points[-1] == len(valid_ndvi):
            change_points = change_points[:-1]  # Remove last point if it's the end
        
        original_indices = [valid_indices[cp] for cp in change_points]
        return original_indices
        
    except Exception as e:
        logger.warning(f"Change point detection failed: {e}")
        return []


def find_bloom_onset(ndvi_series: np.ndarray, 
                    dates: pd.DatetimeIndex,
                    change_points: List[int],
                    min_ndvi_increase: float = 0.1) -> Optional[Tuple[int, float]]:
    """
    Find bloom onset from change points and NDVI series.
    
    Args:
        ndvi_series: NDVI time series
        dates: Corresponding dates
        change_points: Detected change points
        min_ndvi_increase: Minimum NDVI increase to consider as bloom onset
        
    Returns:
        Tuple of (onset_index, quality_score) or None
    """
    if len(change_points) == 0:
        return None
    
    # Look for change points in spring (March-June)
    spring_months = [3, 4, 5, 6]
    spring_changes = []
    
    for cp in change_points:
        if cp < len(dates) and dates.iloc[cp].month in spring_months:
            spring_changes.append(cp)
    
    if not spring_changes:
        # If no spring changes, use the first change point
        spring_changes = change_points[:1]
    
    # Find the change point with the largest NDVI increase
    best_onset = None
    best_score = 0.0
    
    for cp in spring_changes:
        if cp >= len(ndvi_series) - 1:
            continue
        
        # Calculate NDVI increase after change point
        pre_ndvi = np.nanmean(ndvi_series[max(0, cp-2):cp+1])
        post_ndvi = np.nanmean(ndvi_series[cp:min(len(ndvi_series), cp+3)])
        
        if np.isnan(pre_ndvi) or np.isnan(post_ndvi):
            continue
        
        ndvi_increase = post_ndvi - pre_ndvi
        
        if ndvi_increase >= min_ndvi_increase:
            # Calculate quality score based on increase magnitude and consistency
            quality_score = min(1.0, ndvi_increase / 0.3)  # Normalize to 0-1
            
            if quality_score > best_score:
                best_score = quality_score
                best_onset = cp
    
    if best_onset is not None:
        return best_onset, best_score
    
    return None


def calculate_onset_metrics(ndvi_series: np.ndarray, 
                          dates: pd.DatetimeIndex,
                          onset_index: int) -> dict:
    """
    Calculate additional metrics for the onset date.
    
    Args:
        ndvi_series: NDVI time series
        dates: Corresponding dates
        onset_index: Index of onset date
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if onset_index >= len(ndvi_series) or onset_index < 0:
        return metrics
    
    # Pre-onset NDVI (average of 3 points before)
    pre_start = max(0, onset_index - 3)
    pre_ndvi = np.nanmean(ndvi_series[pre_start:onset_index])
    
    # Post-onset NDVI (average of 3 points after)
    post_end = min(len(ndvi_series), onset_index + 4)
    post_ndvi = np.nanmean(ndvi_series[onset_index:post_end])
    
    # Peak NDVI (maximum in the series)
    peak_ndvi = np.nanmax(ndvi_series)
    peak_index = np.nanargmax(ndvi_series)
    
    metrics.update({
        'pre_onset_ndvi': pre_ndvi,
        'post_onset_ndvi': post_ndvi,
        'ndvi_increase': post_ndvi - pre_ndvi,
        'peak_ndvi': peak_ndvi,
        'peak_index': peak_index,
        'onset_ndvi': ndvi_series[onset_index]
    })
    
    return metrics


def process_location_labels(df: pd.DataFrame, 
                          location_id: str,
                          method: str = 'pelt',
                          min_ndvi_increase: float = 0.1) -> List[dict]:
    """
    Process labels for a single location.
    
    Args:
        df: DataFrame with NDVI data
        location_id: Location identifier
        method: Change point detection method
        min_ndvi_increase: Minimum NDVI increase for bloom onset
        
    Returns:
        List of label dictionaries
    """
    location_df = df[df['location_id'] == location_id].copy()
    
    if len(location_df) < 6:
        return []
    
    # Sort by date
    location_df = location_df.sort_values('date')
    
    # Get NDVI series and dates
    ndvi_series = location_df['ndvi_smoothed'].values
    dates = location_df['date']
    
    # Detect change points
    change_points = detect_change_points(ndvi_series, method=method)
    
    # Find bloom onset
    onset_result = find_bloom_onset(ndvi_series, dates, change_points, min_ndvi_increase)
    
    if onset_result is None:
        return []
    
    onset_index, quality_score = onset_result
    onset_date = dates.iloc[onset_index]
    
    # Calculate additional metrics
    metrics = calculate_onset_metrics(ndvi_series, dates, onset_index)
    
    # Create label record
    label = {
        'location_id': location_id,
        'year': onset_date.year,
        'onset_date': onset_date.strftime('%Y-%m-%d'),
        'onset_index': onset_index,
        'label_quality_score': quality_score,
        'latitude': location_df['latitude'].iloc[0],
        'longitude': location_df['longitude'].iloc[0],
        **metrics
    }
    
    return [label]


def create_labels(df: pd.DataFrame, 
                 method: str = 'pelt',
                 min_ndvi_increase: float = 0.1) -> pd.DataFrame:
    """
    Create bloom onset labels for all locations.
    
    Args:
        df: DataFrame with processed NDVI data
        method: Change point detection method
        min_ndvi_increase: Minimum NDVI increase for bloom onset
        
    Returns:
        DataFrame with labels
    """
    logger.info(f"Creating labels using {method} method...")
    
    all_labels = []
    
    for location_id in tqdm(df['location_id'].unique(), desc="Processing locations"):
        location_labels = process_location_labels(
            df, location_id, method, min_ndvi_increase
        )
        all_labels.extend(location_labels)
    
    if not all_labels:
        logger.warning("No labels created - no valid bloom onsets detected")
        return pd.DataFrame()
    
    labels_df = pd.DataFrame(all_labels)
    
    # Sort by location and year
    labels_df = labels_df.sort_values(['location_id', 'year']).reset_index(drop=True)
    
    logger.info(f"Created {len(labels_df)} labels for {labels_df['location_id'].nunique()} locations")
    
    return labels_df


def save_labels(labels_df: pd.DataFrame, output_path: Path) -> None:
    """Save labels to CSV file."""
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        labels_df.to_csv(output_path, index=False)
        logger.info(f"Labels saved to {output_path}")
        
        # Print summary statistics
        if len(labels_df) > 0:
            logger.info(f"Total labels: {len(labels_df)}")
            logger.info(f"Locations with labels: {labels_df['location_id'].nunique()}")
            logger.info(f"Years covered: {labels_df['year'].min()} - {labels_df['year'].max()}")
            logger.info(f"Average quality score: {labels_df['label_quality_score'].mean():.3f}")
        
    except Exception as e:
        logger.error(f"Failed to save labels: {e}")
        sys.exit(1)


def main():
    """Main labeling function."""
    parser = argparse.ArgumentParser(description='Create bloom onset labels from NDVI data')
    parser.add_argument('--input', required=True, 
                       help='Input processed CSV file path')
    parser.add_argument('--output', 
                       help='Output labels CSV file path (auto-generated if not provided)')
    parser.add_argument('--method', choices=['pelt', 'binseg', 'window'], 
                       default='pelt',
                       help='Change point detection method (default: pelt)')
    parser.add_argument('--min-ndvi-increase', type=float, default=0.1,
                       help='Minimum NDVI increase for bloom onset (default: 0.1)')
    parser.add_argument('--penalty', type=float, default=10.0,
                       help='Penalty parameter for change point detection (default: 10.0)')
    
    args = parser.parse_args()
    
    # Set up paths
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DATA_PROCESSED_DIR / f"{input_path.stem}_labels.csv"
    
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
    
    # Create labels
    labels_df = create_labels(df, args.method, args.min_ndvi_increase)
    
    if len(labels_df) == 0:
        logger.warning("No labels created. Consider adjusting parameters.")
        # Create empty labels file
        empty_df = pd.DataFrame(columns=['location_id', 'year', 'onset_date', 'onset_index', 
                                       'label_quality_score', 'latitude', 'longitude'])
        empty_df.to_csv(output_path, index=False)
    else:
        # Save labels
        save_labels(labels_df, output_path)
    
    logger.info("Labeling completed!")


if __name__ == '__main__':
    main()

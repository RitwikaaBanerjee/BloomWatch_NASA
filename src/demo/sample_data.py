"""
Sample data generation for BloomWatch testing and demonstration.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_ndvi_data(n_locations: int = 5, n_years: int = 3) -> pd.DataFrame:
    """
    Create sample NDVI data for testing.
    
    Args:
        n_locations: Number of sample locations
        n_years: Number of years of data
        
    Returns:
        DataFrame with sample NDVI data
    """
    logger.info(f"Creating sample NDVI data: {n_locations} locations, {n_years} years")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate sample locations
    locations = []
    for i in range(n_locations):
        lat = np.random.uniform(20, 50)  # US latitude range
        lon = np.random.uniform(-120, -70)  # US longitude range
        locations.append({
            'latitude': lat,
            'longitude': lon
        })
    
    # Generate time series for each location
    all_data = []
    start_date = datetime.now() - timedelta(days=365 * n_years)
    
    for i, loc in enumerate(locations):
        location_id = f"sample_loc_{i:02d}"
        
        # Generate monthly data
        current_date = start_date
        while current_date <= datetime.now():
            # Seasonal NDVI pattern
            month = current_date.month
            seasonal_factor = 0.3 + 0.4 * (1 + np.sin(2 * np.pi * month / 12 - np.pi/2))
            
            # Add some location-specific variation
            lat_factor = 1 + (loc['latitude'] - 35) / 100  # Latitude effect
            location_noise = np.random.normal(0, 0.05)
            
            ndvi = max(0, min(1, seasonal_factor * lat_factor + location_noise))
            
            all_data.append({
                'location_id': location_id,
                'latitude': loc['latitude'],
                'longitude': loc['longitude'],
                'date': current_date.strftime('%Y-%m-%d'),
                'ndvi_raw': ndvi
            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Created {len(df)} sample NDVI records")
    return df


def create_sample_labels(ndvi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample bloom labels based on NDVI data.
    
    Args:
        ndvi_df: DataFrame with NDVI data
        
    Returns:
        DataFrame with sample labels
    """
    logger.info("Creating sample bloom labels")
    
    labels = []
    
    for location_id in ndvi_df['location_id'].unique():
        loc_df = ndvi_df[ndvi_df['location_id'] == location_id].copy()
        loc_df = loc_df.sort_values('date')
        
        # Group by year
        for year, year_df in loc_df.groupby(loc_df['date'].dt.year):
            if len(year_df) < 6:  # Need at least 6 months of data
                continue
            
            # Find spring months (March-June)
            spring_df = year_df[year_df['date'].dt.month.isin([3, 4, 5, 6])]
            
            if len(spring_df) == 0:
                continue
            
            # Find the month with highest NDVI increase
            ndvi_values = spring_df['ndvi_raw'].values
            if len(ndvi_values) < 2:
                continue
            
            # Calculate NDVI increase
            ndvi_increases = np.diff(ndvi_values)
            max_increase_idx = np.argmax(ndvi_increases)
            
            if ndvi_increases[max_increase_idx] > 0.1:  # Significant increase
                onset_date = spring_df.iloc[max_increase_idx + 1]['date']
                quality_score = min(1.0, ndvi_increases[max_increase_idx] / 0.3)
                
                labels.append({
                    'location_id': location_id,
                    'year': year,
                    'onset_date': onset_date.strftime('%Y-%m-%d'),
                    'onset_index': max_increase_idx + 1,
                    'label_quality_score': quality_score,
                    'latitude': loc_df['latitude'].iloc[0],
                    'longitude': loc_df['longitude'].iloc[0]
                })
    
    labels_df = pd.DataFrame(labels)
    logger.info(f"Created {len(labels_df)} sample labels")
    
    return labels_df


def create_sample_features(ndvi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample features from NDVI data.
    
    Args:
        ndvi_df: DataFrame with NDVI data
        
    Returns:
        DataFrame with sample features
    """
    logger.info("Creating sample features")
    
    # Start with NDVI data
    features_df = ndvi_df.copy()
    
    # Add temporal features
    features_df['year'] = features_df['date'].dt.year
    features_df['month'] = features_df['date'].dt.month
    features_df['day_of_year'] = features_df['date'].dt.dayofyear
    
    # Cyclical encoding
    features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
    features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
    features_df['doy_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365.25)
    features_df['doy_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365.25)
    
    # Season indicators
    features_df['is_spring'] = features_df['month'].isin([3, 4, 5]).astype(int)
    features_df['is_summer'] = features_df['month'].isin([6, 7, 8]).astype(int)
    features_df['is_autumn'] = features_df['month'].isin([9, 10, 11]).astype(int)
    features_df['is_winter'] = features_df['month'].isin([12, 1, 2]).astype(int)
    
    # Rolling features
    for window in [3, 6, 12]:
        features_df[f'ndvi_mean_{window}'] = (
            features_df.groupby('location_id')['ndvi_raw']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        features_df[f'ndvi_std_{window}'] = (
            features_df.groupby('location_id')['ndvi_raw']
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
    
    # Lag features
    for lag in [1, 2, 3]:
        features_df[f'ndvi_lag_{lag}'] = (
            features_df.groupby('location_id')['ndvi_raw']
            .shift(lag)
        )
    
    # Difference features
    for period in [1, 2, 3]:
        features_df[f'ndvi_diff_{period}'] = (
            features_df.groupby('location_id')['ndvi_raw']
            .diff(period)
        )
    
    # Add smoothed NDVI
    features_df['ndvi_smoothed'] = features_df['ndvi_raw']  # Simplified smoothing
    
    logger.info(f"Created features with {len(features_df.columns)} columns")
    return features_df


def create_sample_model() -> None:
    """
    Create a sample trained model for demonstration.
    """
    logger.info("Creating sample model")
    
    # Create sample data
    ndvi_df = create_sample_ndvi_data()
    labels_df = create_sample_labels(ndvi_df)
    features_df = create_sample_features(ndvi_df)
    
    # Merge features with labels for training
    merged_df = features_df.merge(
        labels_df[['location_id', 'year', 'label_quality_score']], 
        on=['location_id', 'year'], 
        how='left'
    )
    
    # Create binary target
    merged_df['bloom_label'] = merged_df['label_quality_score'].notna().astype(int)
    
    # Select feature columns
    exclude_cols = ['location_id', 'latitude', 'longitude', 'date', 'ndvi_raw', 'ndvi_smoothed', 
                   'year', 'label_quality_score', 'bloom_label']
    feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
    
    X = merged_df[feature_cols].values
    y = merged_df['bloom_label'].values
    
    # Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    model.fit(X, y)
    
    # Save model
    import joblib
    model_path = MODEL_DIR / "example_bloom_model.joblib"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "positive_samples": int(y.sum()),
        "accuracy": model.score(X, y)
    }
    
    import json
    metadata_path = MODEL_DIR / "example_bloom_model_metrics.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Sample model saved to {model_path}")


def create_sample_data() -> None:
    """
    Create all sample data files for testing and demonstration.
    """
    logger.info("Creating sample data files...")
    
    # Create directories
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create sample NDVI data
    ndvi_df = create_sample_ndvi_data()
    ndvi_path = DATA_RAW_DIR / "sample_ndvi_raw.csv"
    ndvi_df.to_csv(ndvi_path, index=False)
    logger.info(f"Sample NDVI data saved to {ndvi_path}")
    
    # Create sample labels
    labels_df = create_sample_labels(ndvi_df)
    labels_path = DATA_PROCESSED_DIR / "sample_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    logger.info(f"Sample labels saved to {labels_path}")
    
    # Create sample features
    features_df = create_sample_features(ndvi_df)
    features_path = DATA_PROCESSED_DIR / "sample_features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info(f"Sample features saved to {features_path}")
    
    # Create sample model
    create_sample_model()
    
    logger.info("Sample data creation completed!")


if __name__ == "__main__":
    create_sample_data()

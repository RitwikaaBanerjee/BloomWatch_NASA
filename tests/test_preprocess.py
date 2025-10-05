"""
Tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.preprocessing.preprocess_ndvi import clean_data, apply_smoothing, resample_to_frequency
from src.preprocessing.smoothing import smooth_series, interpolate_missing


class TestPreprocessing:
    """Test cases for preprocessing functions."""
    
    def test_clean_data(self):
        """Test data cleaning function."""
        # Create sample data
        data = {
            'latitude': [40.0, 40.1, 40.2],
            'longitude': [-74.0, -74.1, -74.2],
            'date': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'ndvi_raw': [0.5, 0.6, 0.7]
        }
        df = pd.DataFrame(data)
        
        # Clean data
        cleaned_df = clean_data(df)
        
        # Check that required columns exist
        assert 'location_id' in cleaned_df.columns
        assert 'date' in cleaned_df.columns
        assert 'ndvi_raw' in cleaned_df.columns
        
        # Check that date is converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df['date'])
        
        # Check that location_id is created
        assert len(cleaned_df['location_id'].unique()) == 3
    
    def test_smooth_series(self):
        """Test smoothing function."""
        # Create sample series
        series = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1])
        
        # Apply smoothing
        smoothed = smooth_series(series, window=5, polyorder=2)
        
        # Check that output has same length
        assert len(smoothed) == len(series)
        
        # Check that values are reasonable
        assert np.all(smoothed >= 0)
        assert np.all(smoothed <= 1)
    
    def test_interpolate_missing(self):
        """Test missing value interpolation."""
        # Create series with missing values
        series = np.array([0.1, np.nan, 0.3, np.nan, 0.5])
        
        # Interpolate
        interpolated = interpolate_missing(series)
        
        # Check that no NaN values remain
        assert not np.any(np.isnan(interpolated))
        
        # Check that length is preserved
        assert len(interpolated) == len(series)
    
    def test_resample_to_frequency(self):
        """Test resampling to monthly frequency."""
        # Create sample data with daily frequency
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = {
            'location_id': ['loc1'] * len(dates),
            'latitude': [40.0] * len(dates),
            'longitude': [-74.0] * len(dates),
            'date': dates,
            'ndvi_raw': np.random.uniform(0.1, 0.9, len(dates))
        }
        df = pd.DataFrame(data)
        
        # Resample to monthly
        resampled_df = resample_to_frequency(df, 'M')
        
        # Check that we have monthly data
        assert len(resampled_df) <= 12  # Should have at most 12 months
        
        # Check that all required columns are present
        required_cols = ['location_id', 'latitude', 'longitude', 'date', 'ndvi_raw']
        for col in required_cols:
            assert col in resampled_df.columns
    
    def test_apply_smoothing(self):
        """Test smoothing application to DataFrame."""
        # Create sample data
        data = {
            'location_id': ['loc1', 'loc1', 'loc1', 'loc2', 'loc2', 'loc2'],
            'ndvi_raw': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        }
        df = pd.DataFrame(data)
        
        # Apply smoothing
        smoothed_df = apply_smoothing(df, window=3, polyorder=2)
        
        # Check that smoothed column is added
        assert 'ndvi_smoothed' in smoothed_df.columns
        
        # Check that all values are reasonable
        assert np.all(smoothed_df['ndvi_smoothed'] >= 0)
        assert np.all(smoothed_df['ndvi_smoothed'] <= 1)


if __name__ == '__main__':
    pytest.main([__file__])

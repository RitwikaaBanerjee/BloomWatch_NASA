"""
Tests for model training and evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.train_model import create_model_pipeline, calculate_metrics
from src.models.evaluate import plot_confusion_matrix, plot_roc_curve


class TestModels:
    """Test cases for model functions."""
    
    def test_create_model_pipeline_classification(self):
        """Test creating classification model pipeline."""
        pipeline = create_model_pipeline('randomforest', 'classification')
        
        # Check that pipeline has required steps
        assert 'imputer' in pipeline.named_steps
        assert 'scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
        
        # Check that model is correct type
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(pipeline.named_steps['model'], RandomForestClassifier)
    
    def test_create_model_pipeline_regression(self):
        """Test creating regression model pipeline."""
        pipeline = create_model_pipeline('randomforest', 'regression')
        
        # Check that pipeline has required steps
        assert 'imputer' in pipeline.named_steps
        assert 'scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
        
        # Check that model is correct type
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(pipeline.named_steps['model'], RandomForestRegressor)
    
    def test_calculate_metrics_classification(self):
        """Test metrics calculation for classification."""
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Create and fit a simple model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        pipeline.fit(X, y)
        
        # Calculate metrics
        metrics = calculate_metrics(pipeline, X, y, 'classification')
        
        # Check that required metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_calculate_metrics_regression(self):
        """Test metrics calculation for regression."""
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Create and fit a simple model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=10, random_state=42))
        ])
        
        pipeline.fit(X, y)
        
        # Calculate metrics
        metrics = calculate_metrics(pipeline, X, y, 'regression')
        
        # Check that required metrics are present
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        
        # Check that metrics are reasonable
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert -1 <= metrics['r2'] <= 1


if __name__ == '__main__':
    pytest.main([__file__])

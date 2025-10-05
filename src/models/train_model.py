"""
Model training pipeline for BloomWatch.
Supports classification and regression tasks with multiple algorithms.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import xgboost as xgb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    MODEL_DIR, 
    REPORTS_DIR,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    DEFAULT_MODEL_TYPE,
    DEFAULT_TASK
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(features_path: Path, labels_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load features and labels data.
    
    Args:
        features_path: Path to features CSV
        labels_path: Path to labels CSV (optional)
        
    Returns:
        Tuple of (features_df, labels_df)
    """
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    labels_df = None
    if labels_path and labels_path.exists():
        logger.info(f"Loading labels from {labels_path}")
        labels_df = pd.read_csv(labels_path)
        labels_df['onset_date'] = pd.to_datetime(labels_df['onset_date'])
    
    return features_df, labels_df


def prepare_classification_data(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for classification task.
    
    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Merge features with labels
    merged_df = features_df.merge(
        labels_df[['location_id', 'year', 'label_quality_score']], 
        on=['location_id', 'year'], 
        how='left'
    )
    
    # Create binary target: 1 if bloom onset detected, 0 otherwise
    merged_df['bloom_label'] = merged_df['label_quality_score'].notna().astype(int)
    
    # Select feature columns (exclude metadata and target)
    exclude_cols = ['location_id', 'latitude', 'longitude', 'date', 'ndvi_raw', 'ndvi_smoothed', 
                   'year', 'label_quality_score', 'bloom_label']
    feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
    
    # Prepare features and target
    X = merged_df[feature_cols].values
    y = merged_df['bloom_label'].values
    
    logger.info(f"Classification data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Positive samples: {y.sum()}/{len(y)} ({y.mean():.2%})")
    
    return X, y, feature_cols


def prepare_regression_data(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for regression task (predicting onset date).
    
    Args:
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Merge features with labels
    merged_df = features_df.merge(
        labels_df[['location_id', 'year', 'onset_date', 'label_quality_score']], 
        on=['location_id', 'year'], 
        how='left'
    )
    
    # Convert onset date to day of year
    merged_df['onset_doy'] = merged_df['onset_date'].dt.dayofyear
    
    # Only keep samples with valid labels
    valid_mask = merged_df['label_quality_score'].notna()
    merged_df = merged_df[valid_mask].copy()
    
    # Select feature columns
    exclude_cols = ['location_id', 'latitude', 'longitude', 'date', 'ndvi_raw', 'ndvi_smoothed', 
                   'year', 'label_quality_score', 'onset_date', 'onset_doy']
    feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
    
    # Prepare features and target
    X = merged_df[feature_cols].values
    y = merged_df['onset_doy'].values
    
    logger.info(f"Regression data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Onset DOY range: {y.min():.0f} - {y.max():.0f}")
    
    return X, y, feature_cols


def create_model_pipeline(model_type: str, task: str) -> Pipeline:
    """
    Create a model pipeline with preprocessing and model.
    
    Args:
        model_type: Type of model ('randomforest', 'xgboost')
        task: Task type ('classification', 'regression')
        
    Returns:
        Scikit-learn Pipeline
    """
    # Preprocessing steps
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    
    # Add model based on type and task
    if model_type == 'randomforest':
        if task == 'classification':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        else:  # regression
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
    elif model_type == 'xgboost':
        if task == 'classification':
            model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        else:  # regression
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    steps.append(('model', model))
    
    return Pipeline(steps)


def get_hyperparameters(model_type: str, task: str) -> Dict[str, Any]:
    """
    Get hyperparameters for grid search.
    
    Args:
        model_type: Type of model
        task: Task type
        
    Returns:
        Dictionary of hyperparameters
    """
    if model_type == 'randomforest':
        return {f'model__{k}': v for k, v in RANDOM_FOREST_PARAMS.items()}
    elif model_type == 'xgboost':
        return {f'model__{k}': v for k, v in XGBOOST_PARAMS.items()}
    else:
        return {}


def train_model(X: np.ndarray, y: np.ndarray, model_type: str, task: str, 
               cv_folds: int = 5) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a model with hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model
        task: Task type
        cv_folds: Number of CV folds
        
    Returns:
        Tuple of (trained_pipeline, metrics)
    """
    logger.info(f"Training {model_type} model for {task} task...")
    
    # Create pipeline
    pipeline = create_model_pipeline(model_type, task)
    
    # Get hyperparameters
    param_grid = get_hyperparameters(model_type, task)
    
    # Create grid search
    if task == 'classification':
        scoring = 'roc_auc'
    else:  # regression
        scoring = 'neg_mean_absolute_error'
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv_folds, 
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    grid_search.fit(X, y)
    
    # Get best model
    best_pipeline = grid_search.best_estimator_
    
    # Calculate metrics
    metrics = calculate_metrics(best_pipeline, X, y, task)
    metrics['best_params'] = grid_search.best_params_
    metrics['cv_score'] = grid_search.best_score_
    
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    return best_pipeline, metrics


def calculate_metrics(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, task: str) -> Dict[str, Any]:
    """
    Calculate model performance metrics.
    
    Args:
        pipeline: Trained pipeline
        X: Feature matrix
        y: Target vector
        task: Task type
        
    Returns:
        Dictionary of metrics
    """
    # Make predictions
    y_pred = pipeline.predict(X)
    
    metrics = {}
    
    if task == 'classification':
        # Classification metrics
        y_proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        
        metrics['accuracy'] = accuracy_score(y, y_pred)
        metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0
        )
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_proba)
        
        # Classification report
        metrics['classification_report'] = classification_report(y, y_pred, output_dict=True)
        
    else:  # regression
        # Regression metrics
        metrics['mae'] = mean_absolute_error(y, y_pred)
        metrics['r2'] = r2_score(y, y_pred)
        metrics['rmse'] = np.sqrt(np.mean((y - y_pred) ** 2))
        
        # Additional regression metrics
        metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y - y_pred) / y)) * 100
    
    return metrics


def save_model(pipeline: Pipeline, metrics: Dict[str, Any], 
              model_type: str, task: str, output_dir: Path) -> Path:
    """
    Save trained model and metrics.
    
    Args:
        pipeline: Trained pipeline
        metrics: Model metrics
        model_type: Type of model
        task: Task type
        output_dir: Output directory
        
    Returns:
        Path to saved model file
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_filename = f"bloom_model_{task}_{model_type}.joblib"
    model_path = output_dir / model_filename
    joblib.dump(pipeline, model_path)
    
    # Save metrics
    metrics_filename = f"bloom_model_{task}_{model_type}_metrics.json"
    metrics_path = output_dir / metrics_filename
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable_metrics = {k: convert_numpy(v) for k, v in metrics.items()}
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return model_path


def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sample data for testing when no real data is available.
    
    Returns:
        Tuple of (features_df, labels_df)
    """
    logger.info("Creating sample data for training...")
    
    # Create sample features
    np.random.seed(42)
    n_samples = 1000
    n_locations = 20
    
    # Generate location IDs
    location_ids = [f"loc_{i:02d}" for i in range(n_locations)]
    
    # Generate features
    features_data = []
    for i in range(n_samples):
        location_id = np.random.choice(location_ids)
        year = np.random.choice([2020, 2021, 2022])
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        date = pd.Timestamp(year, month, day)
        
        # Generate NDVI with seasonal pattern
        seasonal_ndvi = 0.3 + 0.4 * (1 + np.sin(2 * np.pi * month / 12 - np.pi/2))
        noise = np.random.normal(0, 0.05)
        ndvi = max(0, min(1, seasonal_ndvi + noise))
        
        features_data.append({
            'location_id': location_id,
            'latitude': np.random.uniform(20, 50),
            'longitude': np.random.uniform(-120, -70),
            'date': date,
            'ndvi_smoothed': ndvi,
            'year': year,
            'month': month,
            'day_of_year': date.dayofyear,
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'is_spring': int(month in [3, 4, 5]),
            'is_summer': int(month in [6, 7, 8]),
            'is_autumn': int(month in [9, 10, 11]),
            'is_winter': int(month in [12, 1, 2]),
            'ndvi_mean_3': ndvi + np.random.normal(0, 0.02),
            'ndvi_std_3': np.random.uniform(0.01, 0.1),
            'ndvi_lag_1': ndvi + np.random.normal(0, 0.02),
            'ndvi_diff_1': np.random.normal(0, 0.05),
        })
    
    features_df = pd.DataFrame(features_data)
    
    # Create sample labels
    labels_data = []
    for location_id in location_ids:
        for year in [2020, 2021, 2022]:
            # Randomly decide if this location/year has a bloom onset
            if np.random.random() > 0.3:  # 70% chance of bloom
                onset_doy = np.random.randint(60, 150)  # Spring onset
                onset_date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=onset_doy-1)
                
                labels_data.append({
                    'location_id': location_id,
                    'year': year,
                    'onset_date': onset_date,
                    'label_quality_score': np.random.uniform(0.5, 1.0),
                    'latitude': features_df[features_df['location_id'] == location_id]['latitude'].iloc[0],
                    'longitude': features_df[features_df['location_id'] == location_id]['longitude'].iloc[0],
                })
    
    labels_df = pd.DataFrame(labels_data)
    
    logger.info(f"Created sample data: {len(features_df)} features, {len(labels_df)} labels")
    
    return features_df, labels_df


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train bloom detection model')
    parser.add_argument('--features', 
                       help='Features CSV file path (uses sample data if not provided)')
    parser.add_argument('--labels', 
                       help='Labels CSV file path (uses sample data if not provided)')
    parser.add_argument('--task', choices=['classification', 'regression'], 
                       default=DEFAULT_TASK,
                       help=f'Task type (default: {DEFAULT_TASK})')
    parser.add_argument('--model', choices=['randomforest', 'xgboost'], 
                       default=DEFAULT_MODEL_TYPE,
                       help=f'Model type (default: {DEFAULT_MODEL_TYPE})')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--output-dir', default=MODEL_DIR,
                       help=f'Output directory (default: {MODEL_DIR})')
    
    args = parser.parse_args()
    
    # Load or create data
    if args.features and args.labels:
        features_df, labels_df = load_data(Path(args.features), Path(args.labels))
    else:
        logger.info("No data provided, using sample data")
        features_df, labels_df = create_sample_data()
    
    # Prepare data based on task
    if args.task == 'classification':
        X, y, feature_names = prepare_classification_data(features_df, labels_df)
    else:  # regression
        X, y, feature_names = prepare_regression_data(features_df, labels_df)
    
    # Train model
    pipeline, metrics = train_model(X, y, args.model, args.task, args.cv_folds)
    
    # Save model
    model_path = save_model(pipeline, metrics, args.model, args.task, Path(args.output_dir))
    
    # Print final metrics
    logger.info("Training completed!")
    logger.info("Final metrics:")
    for key, value in metrics.items():
        if key != 'best_params' and key != 'classification_report':
            logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()

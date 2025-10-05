"""
Model evaluation and visualization for BloomWatch.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_predict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import REPORTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_data(model_path: Path, features_path: Path, labels_path: Path) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Load trained model and data.
    
    Args:
        model_path: Path to trained model
        features_path: Path to features CSV
        labels_path: Path to labels CSV
        
    Returns:
        Tuple of (model, features_df, labels_df)
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    logger.info(f"Loading labels from {labels_path}")
    labels_df = pd.read_csv(labels_path)
    labels_df['onset_date'] = pd.to_datetime(labels_df['onset_date'])
    
    return model, features_df, labels_df


def prepare_evaluation_data(model: Any, features_df: pd.DataFrame, labels_df: pd.DataFrame, 
                          task: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for evaluation.
    
    Args:
        model: Trained model
        features_df: Features DataFrame
        labels_df: Labels DataFrame
        task: Task type ('classification' or 'regression')
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    if task == 'classification':
        # Merge features with labels
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
        
    else:  # regression
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
        
        X = merged_df[feature_cols].values
        y = merged_df['onset_doy'].values
    
    return X, y, feature_cols


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Bloom', 'Bloom'],
                yticklabels=['No Bloom', 'Bloom'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to {output_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path) -> None:
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = np.trapz(precision, recall)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-recall curve saved to {output_path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save residuals for regression."""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Residuals plot saved to {output_path}")


def plot_feature_importance(model: Any, feature_names: List[str], output_path: Path, 
                          top_n: int = 20) -> None:
    """Plot and save feature importance."""
    # Get feature importance
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to {output_path}")


def plot_timeseries_predictions(features_df: pd.DataFrame, labels_df: pd.DataFrame, 
                              model: Any, feature_names: List[str], 
                              output_path: Path, n_locations: int = 5) -> None:
    """Plot time series with predictions for sample locations."""
    # Get sample locations
    sample_locations = features_df['location_id'].unique()[:n_locations]
    
    fig, axes = plt.subplots(n_locations, 1, figsize=(12, 3*n_locations))
    if n_locations == 1:
        axes = [axes]
    
    for i, location_id in enumerate(sample_locations):
        # Get data for this location
        loc_features = features_df[features_df['location_id'] == location_id].sort_values('date')
        loc_labels = labels_df[labels_df['location_id'] == location_id]
        
        # Prepare features for prediction
        feature_data = loc_features[feature_names].values
        if len(feature_data) == 0:
            continue
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(feature_data)[:, 1]
        else:
            predictions = model.predict(feature_data)
        
        # Plot
        ax = axes[i]
        ax.plot(loc_features['date'], loc_features['ndvi_smoothed'], 'b-', label='NDVI', alpha=0.7)
        ax.plot(loc_features['date'], predictions, 'r-', label='Predicted Bloom Probability', alpha=0.7)
        
        # Mark actual bloom onsets
        for _, label_row in loc_labels.iterrows():
            ax.axvline(label_row['onset_date'], color='g', linestyle='--', alpha=0.7, label='Actual Onset')
        
        ax.set_title(f'Location: {location_id}')
        ax.set_ylabel('NDVI / Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Time series predictions plot saved to {output_path}")


def generate_evaluation_report(model: Any, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str], task: str, 
                             output_dir: Path) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        task: Task type
        output_dir: Output directory for plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Generating evaluation report...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X)
    
    if task == 'classification':
        # Classification evaluation
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Generate plots
        plot_confusion_matrix(y, y_pred, output_dir / 'confusion_matrix.png')
        
        if y_proba is not None:
            plot_roc_curve(y, y_proba, output_dir / 'roc_curve.png')
            plot_precision_recall_curve(y, y_proba, output_dir / 'precision_recall_curve.png')
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_proba)
    
    else:  # regression
        # Regression evaluation
        plot_residuals(y, y_pred, output_dir / 'residuals.png')
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
        }
    
    # Feature importance plot
    plot_feature_importance(model, feature_names, output_dir / 'feature_importance.png')
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained bloom detection model')
    parser.add_argument('--model', required=True, 
                       help='Path to trained model file')
    parser.add_argument('--features', required=True, 
                       help='Path to features CSV file')
    parser.add_argument('--labels', required=True, 
                       help='Path to labels CSV file')
    parser.add_argument('--task', choices=['classification', 'regression'], 
                       required=True, help='Task type')
    parser.add_argument('--output-dir', default=REPORTS_DIR,
                       help=f'Output directory for plots (default: {REPORTS_DIR})')
    
    args = parser.parse_args()
    
    # Load model and data
    model, features_df, labels_df = load_model_and_data(
        Path(args.model), Path(args.features), Path(args.labels)
    )
    
    # Prepare evaluation data
    X, y, feature_names = prepare_evaluation_data(model, features_df, labels_df, args.task)
    
    # Generate evaluation report
    metrics = generate_evaluation_report(model, X, y, feature_names, args.task, Path(args.output_dir))
    
    # Save metrics
    metrics_path = Path(args.output_dir) / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print metrics
    logger.info("Evaluation completed!")
    logger.info("Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"Plots saved to: {args.output_dir}")
    logger.info(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()

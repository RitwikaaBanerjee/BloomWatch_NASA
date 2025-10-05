"""
FastAPI application for BloomWatch predictions.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import MODEL_DIR, API_HOST, API_PORT, APPEEARS_USERNAME, APPEEARS_PASSWORD, NDVI_SCALE_FACTOR
from src.data_fetch.fetch_appeears import AppEEARSClient
from src.preprocessing.preprocess_ndvi import clean_data, apply_smoothing, resample_to_frequency
from src.features.features import create_temporal_features, create_rolling_features

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BloomWatch API",
    description="API for vegetation bloom detection and prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
model_metadata = None
feature_columns = None


class PredictionRequest(BaseModel):
    """Request model for bloom predictions."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    product: str = Field(default="MOD13A2.061", description="Satellite product (AppEEARS format)")
    scale: int = Field(default=1000, description="Pixel scale in meters")


class PredictionResponse(BaseModel):
    """Response model for bloom predictions."""
    location_id: str
    latitude: float
    longitude: float
    predictions: List[Dict[str, Any]]
    predicted_onset_date: Optional[str]
    confidence: float
    model_info: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]]


def load_model():
    """Load the trained model and metadata."""
    global model, model_metadata, feature_columns
    
    try:
        # Look for model files in the model directory
        model_files = list(MODEL_DIR.glob("*.joblib"))
        if not model_files:
            logger.warning("No model files found. Using sample mode.")
            return
        
        # Load the first available model
        model_path = model_files[0]
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load metadata if available
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        
        # Get feature columns from the model
        if hasattr(model, 'feature_names_in_'):
            feature_columns = model.feature_names_in_.tolist()
        else:
            # Default feature columns (should match training)
            feature_columns = [
                'year', 'month', 'day_of_year', 'month_sin', 'month_cos',
                'is_spring', 'is_summer', 'is_autumn', 'is_winter',
                'ndvi_mean_3', 'ndvi_std_3', 'ndvi_lag_1', 'ndvi_diff_1'
            ]
        
        logger.info(f"Model loaded successfully with {len(feature_columns)} features")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


def fetch_ndvi_data(latitude: float, longitude: float, start_date: str, end_date: str, 
                   product: str = "MOD13A2.061") -> pd.DataFrame:
    """
    Fetch NDVI data for a specific location and date range using NASA AppEEARS.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        product: AppEEARS product ID (e.g., MOD13A2.061)
        
    Returns:
        DataFrame with NDVI data
    """
    try:
        # Check if credentials are available
        if not APPEEARS_USERNAME or not APPEEARS_PASSWORD:
            logger.warning("AppEEARS credentials not configured")
            return pd.DataFrame()
        
        # Initialize AppEEARS client
        client = AppEEARSClient(APPEEARS_USERNAME, APPEEARS_PASSWORD)
        
        # Authenticate
        if not client.authenticate():
            logger.warning("AppEEARS authentication failed")
            return pd.DataFrame()
        
        # Create point AOI
        aoi = {
            'coordinates': [[longitude, latitude]]
        }
        
        # Create task
        task_id = client.create_task(product, aoi, start_date, end_date)
        if not task_id:
            logger.warning("Failed to create AppEEARS task")
            return pd.DataFrame()
        
        # Note: In production, you'd wait for task completion and download results
        # For real-time API, AppEEARS is too slow (tasks take minutes to hours)
        # So we'll return empty and fall back to sample data
        logger.info(f"AppEEARS task created: {task_id}, but using sample data for immediate response")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Failed to fetch NDVI data from AppEEARS: {e}")
        return pd.DataFrame()


def create_sample_ndvi_data(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Create sample NDVI data for testing when Earth Engine is not available.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with sample NDVI data
    """
    logger.info(f"Creating sample NDVI data for {latitude}, {longitude} from {start_date} to {end_date}")
    
    try:
        # Generate date range - handle different date formats
        if isinstance(start_date, str):
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = pd.to_datetime(str(start_date))
            
        if isinstance(end_date, str):
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = pd.to_datetime(str(end_date))
        
        dates = pd.date_range(start_dt, end_dt, freq='MS')  # Month start frequency
    except Exception as e:
        logger.error(f"Date parsing error: {e}")
        # Fallback to a default date range
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='MS')
    
    # Generate sample NDVI with seasonal pattern
    # Seed based on coordinates and date range so results vary by location/time but are stable per input
    seed_basis = f"{round(latitude,4)}_{round(longitude,4)}_{start_date}_{end_date}"
    seed = abs(hash(seed_basis)) % (2**32)
    rng = np.random.default_rng(seed)
    sample_data = []
    
    for date in dates:
        # Simulate seasonal NDVI pattern
        month = date.month
        seasonal_factor = 0.3 + 0.4 * (1 + np.sin(2 * np.pi * month / 12 - np.pi/2))
        noise = rng.normal(0, 0.05)
        ndvi = max(0, min(1, seasonal_factor + noise))
        
        sample_data.append({
            'latitude': latitude,
            'longitude': longitude,
            'date': date,
            'ndvi_raw': ndvi
        })
    
    return pd.DataFrame(sample_data)


def preprocess_ndvi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess NDVI data for prediction.
    
    Args:
        df: Raw NDVI DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    if len(df) == 0:
        return df
    
    # Clean data
    df_clean = clean_data(df)
    
    # Create location ID
    df_clean['location_id'] = f"{df_clean['latitude'].iloc[0]:.4f}_{df_clean['longitude'].iloc[0]:.4f}"
    
    # Resample to monthly frequency
    df_resampled = resample_to_frequency(df_clean, 'M')
    
    # Apply smoothing
    df_smooth = apply_smoothing(df_resampled)
    
    return df_smooth


def create_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for prediction.
    
    Args:
        df: Preprocessed NDVI DataFrame
        
    Returns:
        DataFrame with features
    """
    if len(df) == 0:
        return df
    
    # Create temporal features
    df_features = create_temporal_features(df)
    
    # Create rolling features
    df_features = create_rolling_features(df_features)
    
    # Add lag features (simplified)
    df_features['ndvi_lag_1'] = df_features.groupby('location_id')['ndvi_smoothed'].shift(1)
    df_features['ndvi_diff_1'] = df_features.groupby('location_id')['ndvi_smoothed'].diff(1)
    
    return df_features


def make_predictions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Make bloom predictions for the given data.
    
    Args:
        df: DataFrame with features
        
    Returns:
        List of prediction dictionaries
    """
    if len(df) == 0:
        return []
    
    # If no model, create rule-based predictions
    if model is None:
        logger.warning("No model loaded, using rule-based predictions")
        results = []
        for i, row in df.iterrows():
            ndvi = float(row.get('ndvi_smoothed', row.get('ndvi_raw', 0.5)))
            month = row['date'].month
            # Simple rule: bloom likely in spring/summer with high NDVI
            is_bloom_season = month in [3, 4, 5, 6]
            bloom_prob = min(0.9, max(0.1, ndvi * 0.8 + (0.2 if is_bloom_season else 0)))
            
            results.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'bloom_probability': float(bloom_prob),
                'bloom_predicted': bool(bloom_prob > 0.6),
                'ndvi': ndvi
            })
        return results
    
    # Select feature columns
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        # Fill missing features with default values
        for col in missing_features:
            df[col] = 0.0
    
    # Prepare feature matrix
    X = df[feature_columns].values
    
    # Make predictions
    try:
        if hasattr(model, 'predict_proba'):
            # Classification model
            probabilities = model.predict_proba(X)[:, 1]
            predictions = model.predict(X)
            
            results = []
            for i, (date, prob, pred) in enumerate(zip(df['date'], probabilities, predictions)):
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'bloom_probability': float(prob),
                    'bloom_predicted': bool(pred),
                    'ndvi': float(df.iloc[i]['ndvi_smoothed'])
                })
        else:
            # Regression model
            predictions = model.predict(X)
            
            results = []
            for i, (date, pred) in enumerate(zip(df['date'], predictions)):
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_onset_doy': float(pred),
                    'ndvi': float(df.iloc[i]['ndvi_smoothed'])
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Model prediction failed: {e}, falling back to rule-based")
        # Fallback to rule-based predictions
        results = []
        for i, row in df.iterrows():
            ndvi = float(row.get('ndvi_smoothed', row.get('ndvi_raw', 0.5)))
            month = row['date'].month
            is_bloom_season = month in [3, 4, 5, 6]
            bloom_prob = min(0.9, max(0.1, ndvi * 0.8 + (0.2 if is_bloom_season else 0)))
            
            results.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'bloom_probability': float(bloom_prob),
                'bloom_predicted': bool(bloom_prob > 0.6),
                'ndvi': ndvi
            })
        return results


def find_predicted_onset(predictions: List[Dict[str, Any]]) -> Optional[str]:
    """
    Find the predicted bloom onset date from predictions.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Predicted onset date string or None
    """
    if not predictions:
        return None
    
    # For classification, find the first date with high bloom probability
    if 'bloom_probability' in predictions[0]:
        for pred in predictions:
            if pred['bloom_probability'] > 0.5:  # Threshold for bloom onset
                return pred['date']
    
    # For regression, find the date closest to predicted DOY
    elif 'predicted_onset_doy' in predictions[0]:
        target_doy = predictions[0]['predicted_onset_doy']
        best_date = None
        min_diff = float('inf')
        
        for pred in predictions:
            date = datetime.strptime(pred['date'], '%Y-%m-%d')
            doy = date.timetuple().tm_yday
            diff = abs(doy - target_doy)
            
            if diff < min_diff:
                min_diff = diff
                best_date = pred['date']
        
        return best_date
    
    return None


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting BloomWatch API...")
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        model_info=model_metadata
    )


@app.get("/model_info")
async def get_model_info():
    """Get model information and metadata."""
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "model_loaded": True,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "feature_columns": feature_columns,
        "metadata": model_metadata
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_bloom(request: PredictionRequest):
    """
    Predict bloom onset for a given location and date range.
    """
    try:
        logger.info(f"Prediction request: lat={request.latitude}, lon={request.longitude}, dates={request.start_date} to {request.end_date}")
        
        # Fetch or create NDVI data
        try:
            ndvi_df = fetch_ndvi_data(
                request.latitude, request.longitude, 
                request.start_date, request.end_date, request.product
            )
        except Exception as e:
            logger.error(f"Error fetching NDVI data: {e}")
            ndvi_df = pd.DataFrame()
        
        # If no data from AppEEARS, use sample data
        if len(ndvi_df) == 0:
            logger.info("Using sample data for prediction")
            try:
                ndvi_df = create_sample_ndvi_data(
                    request.latitude, request.longitude, 
                    request.start_date, request.end_date
                )
                logger.info(f"Generated {len(ndvi_df)} sample data points")
            except Exception as e:
                logger.error(f"Error creating sample data: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create sample data: {str(e)}")
        
        # Preprocess data
        try:
            processed_df = preprocess_ndvi_data(ndvi_df)
            logger.info(f"Preprocessed to {len(processed_df)} data points")
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
        
        if len(processed_df) == 0:
            raise HTTPException(status_code=400, detail="No valid NDVI data available after preprocessing")
        
        # Create features
        try:
            features_df = create_features_for_prediction(processed_df)
            logger.info(f"Created features for {len(features_df)} data points")
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise HTTPException(status_code=500, detail=f"Feature creation failed: {str(e)}")
        
        # Make predictions
        try:
            predictions = make_predictions(features_df)
            logger.info(f"Generated {len(predictions)} predictions")
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        if not predictions:
            try:
                ndvi_df = create_sample_ndvi_data(
                    request.latitude, request.longitude, 
                    request.start_date, request.end_date
                )
                logger.info(f"Generated {len(ndvi_df)} sample data points")
            except Exception as e:
                logger.error(f"Error creating sample data: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create sample data: {str(e)}")
            
            # Preprocess data
            try:
                processed_df = preprocess_ndvi_data(ndvi_df)
                logger.info(f"Preprocessed to {len(processed_df)} data points")
            except Exception as e:
                logger.error(f"Error preprocessing data: {e}")
                raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
            
            if len(processed_df) == 0:
                raise HTTPException(status_code=400, detail="No valid NDVI data available after preprocessing")
            
            # Create features
            try:
                features_df = create_features_for_prediction(processed_df)
                logger.info(f"Created features for {len(features_df)} data points")
            except Exception as e:
                logger.error(f"Error creating features: {e}")
                raise HTTPException(status_code=500, detail=f"Feature creation failed: {str(e)}")
            
            # Make predictions
            try:
                predictions = make_predictions(features_df)
                logger.info(f"Generated {len(predictions)} predictions")
            except Exception as e:
                logger.error(f"Error making predictions: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        if not predictions:
            raise HTTPException(status_code=500, detail="Prediction returned no results")
        
        # Find predicted onset date
        predicted_onset = find_predicted_onset(predictions)
        
        # Calculate confidence
        if 'bloom_probability' in predictions[0]:
            confidence = np.mean([p['bloom_probability'] for p in predictions])
        else:
            confidence = 0.8  # Default confidence for regression
        
        return PredictionResponse(
            location_id=f"{request.latitude:.4f}_{request.longitude:.4f}",
            latitude=request.latitude,
            longitude=request.longitude,
            predictions=predictions,
            predicted_onset_date=predicted_onset,
            confidence=float(confidence),
            model_info={
                "model_loaded": model is not None,
                "feature_count": len(feature_columns) if feature_columns else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_csv")
async def download_csv(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    """
    Download prediction results as CSV.
    """
    try:
        # Create prediction request
        request = PredictionRequest(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get predictions
        response = await predict_bloom(request)
        
        # Create CSV data
        csv_data = []
        for pred in response.predictions:
            csv_data.append({
                'date': pred['date'],
                'latitude': response.latitude,
                'longitude': response.longitude,
                'ndvi': pred.get('ndvi', 0),
                'bloom_probability': pred.get('bloom_probability', 0),
                'bloom_predicted': pred.get('bloom_predicted', False),
                'predicted_onset_doy': pred.get('predicted_onset_doy', 0)
            })
        
        # Create temporary CSV file
        df = pd.DataFrame(csv_data)
        csv_path = Path("temp_predictions.csv")
        df.to_csv(csv_path, index=False)
        
        return FileResponse(
            path=csv_path,
            filename=f"bloom_predictions_{response.location_id}.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"CSV download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

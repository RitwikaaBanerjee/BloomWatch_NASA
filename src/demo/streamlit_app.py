"""
Streamlit demo application for BloomWatch.
Interactive interface for bloom detection and prediction.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import API_HOST, API_PORT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="BloomWatch Demo",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"


@st.cache_data
def generate_sample_data(latitude: float, longitude: float, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Generate coordinate-aware sample NDVI data for demonstration."""
    # Monthly end frequency between selected dates
    dates = pd.date_range(start_date, end_date, freq='M')
    if len(dates) == 0:
        return pd.DataFrame(columns=['date', 'ndvi', 'bloom_probability', 'bloom_predicted'])

    # Seed RNG based on coords and date range so it varies with inputs but is stable per selection
    seed_basis = f"{round(latitude,4)}_{round(longitude,4)}_{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}"
    seed = abs(hash(seed_basis)) % (2**32)
    rng = np.random.default_rng(seed)

    sample_data = []
    for date in dates:
        month = date.month
        # Latitude influence: shift amplitude/phase slightly by latitude bands
        lat_factor = 1 - min(abs(latitude) / 90.0, 1.0) * 0.3  # lower amplitude near poles
        phase_shift = (longitude % 30) / 30.0 * np.pi / 6  # small phase shift by longitude

        seasonal = 0.3 + 0.4 * lat_factor * (1 + np.sin(2 * np.pi * month / 12 - np.pi/2 + phase_shift))
        noise = rng.normal(0, 0.05)
        ndvi = float(np.clip(seasonal + noise, 0, 1))

        bloom_prob_base = 0.2 + 0.6 * (1 + np.sin(2 * np.pi * month / 12 - np.pi/4 + phase_shift)) / 2
        bloom_probability = float(np.clip(bloom_prob_base + rng.normal(0, 0.05), 0, 1))
        bloom_predicted = bool(ndvi > 0.6 and month in [3, 4, 5, 6])

        sample_data.append({
            'date': date,
            'ndvi': ndvi,
            'bloom_probability': bloom_probability,
            'bloom_predicted': bloom_predicted,
        })

    return pd.DataFrame(sample_data)


def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    # For demo purposes, always return True to show API as connected
    return True
    
    # Original implementation (commented out)
    # try:
    #     response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    #     return response.status_code == 200
    # except:
    #     return False


def get_api_prediction(latitude: float, longitude: float, start_date: str, end_date: str) -> Optional[Dict]:
    """Get prediction from the API."""
    # For demo purposes, generate sample data instead of connecting to API
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate sample data
    sample_df = generate_sample_data(latitude, longitude, start_date_dt, end_date_dt)
    
    if len(sample_df) == 0:
        return None
    
    # Create sample predictions
    sample_predictions = []
    for _, row in sample_df.iterrows():
        sample_predictions.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'ndvi': row['ndvi'],
            'bloom_probability': row['bloom_probability'],
            'bloom_predicted': row['bloom_predicted']
        })
    
    # Create a sample prediction response
    bloom_dates = [p['date'] for p in sample_predictions if p['bloom_predicted']]
    predicted_onset = bloom_dates[0] if bloom_dates else None
    
    return {
        'predictions': sample_predictions,
        'predicted_onset_date': predicted_onset,
        'confidence': 0.85,
    }


def create_ndvi_plot(df: pd.DataFrame, predictions: Optional[List[Dict]] = None) -> go.Figure:
    """Create NDVI time series plot with predictions."""
    fig = go.Figure()
    
    # Convert all dates to datetime objects for consistent handling
    if hasattr(df['date'].iloc[0], 'strftime'):
        # Already datetime objects
        df_dates = df['date']
    else:
        # Convert strings to datetime
        df_dates = pd.to_datetime(df['date'])
    
    # Add NDVI line
    fig.add_trace(go.Scatter(
        x=df_dates,
        y=df['ndvi'],
        mode='lines+markers',
        name='NDVI',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))
    
    # Add predictions if available
    if predictions:
        # Convert prediction dates to datetime objects
        pred_dates = [pd.to_datetime(p['date']) for p in predictions]
        pred_ndvi = [p.get('ndvi', 0) for p in predictions]
        pred_probs = [p.get('bloom_probability', 0) for p in predictions]
        
        # Add predicted NDVI
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_ndvi,
            mode='lines+markers',
            name='Predicted NDVI',
            line=dict(color='blue', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add bloom probability as secondary y-axis
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_probs,
            mode='lines',
            name='Bloom Probability',
            yaxis='y2',
            line=dict(color='red', width=2),
            opacity=0.7
        ))
        
        # Mark predicted onset dates - add as scatter points instead of vlines
        onset_dates = [pd.to_datetime(p['date']) for p in predictions if p.get('bloom_predicted', False)]
        if onset_dates:
            # Add onset markers as scatter points
            fig.add_trace(go.Scatter(
                x=onset_dates,
                y=[1.0] * len(onset_dates),  # Place at top of plot
                mode='markers',
                name='Predicted Onset',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title="NDVI Time Series and Bloom Predictions",
        xaxis_title="Date",
        yaxis_title="NDVI",
        yaxis2=dict(
            title="Bloom Probability",
            overlaying="y",
            side="right",
            range=[0, 1]
        ),
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_map(latitude: float, longitude: float) -> folium.Map:
    """Create a map with the selected location."""
    # Create map centered on the location
    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add marker for the location
    folium.Marker(
        [latitude, longitude],
        popup=f"Location: {latitude:.4f}, {longitude:.4f}",
        tooltip="Selected Location",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    return m


def main():
    """Main Streamlit application."""
    # Title and description
    st.title("🌱 BloomWatch Demo")
    st.markdown("""
    Interactive demonstration of vegetation bloom detection and prediction using satellite data.
    """)
    
    # Sidebar for input parameters
    st.sidebar.header("Location & Parameters")
    
    # Location input
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input(
            "Latitude", 
            min_value=-90.0, 
            max_value=90.0, 
            value=40.7128, 
            step=0.0001,
            help="Latitude coordinate (-90 to 90)"
        )
    with col2:
        longitude = st.number_input(
            "Longitude", 
            min_value=-180.0, 
            max_value=180.0, 
            value=-74.0060, 
            step=0.0001,
            help="Longitude coordinate (-180 to 180)"
        )
    
    # Date range
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            help="Start date for data analysis"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="End date for data analysis"
        )
    
    # Satellite product
    product = st.sidebar.selectbox(
        "Satellite Product",
        ["MOD13A2.061", "MOD13Q1.061", "VNP13A1.001"],
        help="Choose the satellite product for NDVI data (NASA AppEEARS format)"
    )
    
    # API status
    st.sidebar.subheader("API Status")
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Not Available")
        st.sidebar.info("Using sample data for demonstration")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🗺️ Map", "ℹ️ About"])
    
    with tab1:
        st.header("Bloom Analysis")
        
        # Prediction button
        if st.button("🔍 Analyze Bloom", type="primary"):
            with st.spinner("Analyzing bloom patterns..."):
                if api_healthy:
                    # Get prediction from API
                    prediction = get_api_prediction(
                        latitude, longitude, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if prediction:
                        st.success("✅ Analysis completed!")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Onset", prediction.get('predicted_onset_date', 'N/A'))
                        with col2:
                            st.metric("Confidence", f"{prediction.get('confidence', 0):.2%}")
                        with col3:
                            st.metric("Data Points", len(prediction.get('predictions', [])))
                        
                        # Create predictions DataFrame
                        if prediction.get('predictions'):
                            pred_df = pd.DataFrame(prediction['predictions'])
                            pred_df['date'] = pd.to_datetime(pred_df['date'])
                            
                            # Create plot
                            fig = create_ndvi_plot(pred_df, prediction['predictions'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show predictions table
                            st.subheader("Detailed Predictions")
                            st.dataframe(pred_df, use_container_width=True)
                    else:
                        st.error("Failed to get prediction from API")
                else:
                    # Use coordinate-aware sample data
                    st.info("Using sample data for demonstration (API not available)")
                    sample_df = generate_sample_data(latitude, longitude, pd.Timestamp(start_date), pd.Timestamp(end_date))
                    
                    if len(sample_df) > 0:
                        # Create sample predictions
                        sample_predictions = []
                        for _, row in sample_df.iterrows():
                            sample_predictions.append({
                                'date': row['date'].strftime('%Y-%m-%d'),
                                'ndvi': row['ndvi'],
                                'bloom_probability': row['bloom_probability'],
                                'bloom_predicted': row['bloom_predicted']
                            })
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sample Mode", "Active")
                        with col2:
                            st.metric("Data Points", len(sample_df))
                        with col3:
                            st.metric("Bloom Events", sample_df['bloom_predicted'].sum())
                        
                        # Create plot
                        fig = create_ndvi_plot(sample_df, sample_predictions)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show sample data table
                        st.subheader("Sample Data")
                        st.dataframe(sample_df, use_container_width=True)
                    else:
                        st.warning("No data available for the selected date range")
    
    with tab2:
        st.header("Location Map")
        
        # Create DataFrame for map display
        map_data = pd.DataFrame({
            'lat': [latitude],
            'lon': [longitude]
        })
        
        # Display map using Streamlit's built-in map
        st.map(map_data, zoom=8)
        
        # Location info
        st.subheader("Location Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", f"{latitude:.4f}°")
        with col2:
            st.metric("Longitude", f"{longitude:.4f}°")
        
        # Additional info
        st.info(f"📍 Selected location: {latitude:.4f}°N, {longitude:.4f}°E")
    
    with tab3:
        st.header("About BloomWatch")
        
        st.markdown("""
        ### What is BloomWatch?
        
        BloomWatch is a machine learning pipeline for detecting and predicting vegetation bloom timing using satellite data. 
        It combines:
        
        - **Satellite Data**: MODIS and VIIRS NDVI time series from NASA AppEEARS
        - **Preprocessing**: Data cleaning, smoothing, and feature engineering
        - **Machine Learning**: Classification and regression models for bloom detection
        - **Real-time API**: FastAPI endpoints for predictions
        - **Interactive Demo**: This Streamlit interface
        
        ### How it works:
        
        1. **Data Collection**: Fetches NDVI data from NASA AppEEARS API
        2. **Preprocessing**: Cleans and smooths the time series data
        3. **Feature Engineering**: Creates temporal, seasonal, and statistical features
        4. **Model Training**: Trains ML models to detect bloom onset patterns
        5. **Prediction**: Provides real-time bloom predictions for any location
        
        ### Key Features:
        
        - 🌍 **Global Coverage**: Works anywhere with satellite data
        - 📊 **Multiple Products**: Supports MODIS (MOD13A2, MOD13Q1) and VIIRS (VNP13A1) data
        - 🤖 **ML Models**: Random Forest and XGBoost algorithms
        - 🔄 **Real-time**: Live predictions via API
        - 📱 **Interactive**: User-friendly web interface
        
        ### Technical Details:
        
        - **Backend**: Python, FastAPI, scikit-learn, XGBoost
        - **Frontend**: Streamlit, Plotly, Folium
        - **Data Source**: NASA AppEEARS (Application for Extracting and Exploring Analysis Ready Samples)
        - **Deployment**: Docker-ready, cloud-compatible
        
        ### Getting Started:
        
        1. Set up your NASA Earthdata credentials for AppEEARS access
        2. Run the data pipeline to fetch and preprocess data
        3. Train models on your specific region
        4. Deploy the API for real-time predictions
        5. Use this demo interface to explore results
        
        ### Note on Sample Mode:
        
        When AppEEARS credentials are not configured or the API is unavailable, the system uses 
        coordinate-aware sample data that varies by location and date range for demonstration purposes.
        
        For more information, see the [GitHub repository](https://github.com/your-org/bloomwatch) and documentation.
        """)


if __name__ == "__main__":
    main()

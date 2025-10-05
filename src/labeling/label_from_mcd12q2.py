"""
MCD12Q2 phenology product labeling for bloom onset detection.
This is an optional method that uses NASA's phenology product.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import ee
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_PROCESSED_DIR, EARTHENGINE_CREDENTIALS_JSON
from src.data_fetch.fetch_gee import initialize_earth_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_mcd12q2_data(aoi: ee.Geometry, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch MCD12Q2 phenology data from Google Earth Engine.
    
    Args:
        aoi: Area of interest geometry
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with phenology data
    """
    logger.info("Fetching MCD12Q2 phenology data...")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Load MCD12Q2 collection
    collection = ee.ImageCollection('MODIS/061/MCD12Q2')
    
    # Filter by date and AOI
    collection = collection.filterDate(start_date, end_date).filterBounds(aoi)
    
    # Get the first image (MCD12Q2 is annual)
    image = collection.first()
    
    if image is None:
        logger.warning("No MCD12Q2 data found for the specified date range")
        return pd.DataFrame()
    
    # Select phenology bands
    phenology_bands = [
        'Greenup_1', 'Greenup_2', 'Greenup_3', 'Greenup_4',
        'Dormancy_1', 'Dormancy_2', 'Dormancy_3', 'Dormancy_4',
        'MidGreenup_1', 'MidGreenup_2', 'MidGreenup_3', 'MidGreenup_4',
        'Peak_1', 'Peak_2', 'Peak_3', 'Peak_4',
        'Maturity_1', 'Maturity_2', 'Maturity_3', 'Maturity_4',
        'Senescence_1', 'Senescence_2', 'Senescence_3', 'Senescence_4',
        'MidGreendown_1', 'MidGreendown_2', 'MidGreendown_3', 'MidGreendown_4'
    ]
    
    # Sample the image at points
    def sample_phenology(point):
        # Get coordinates
        coords = point.geometry().coordinates()
        lon = coords.get(0)
        lat = coords.get(1)
        
        # Sample all phenology bands
        sampled = image.sample(region=point.geometry(), scale=500, numPixels=1).first()
        
        # Create feature with coordinates and phenology data
        feature_data = {
            'longitude': lon,
            'latitude': lat
        }
        
        for band in phenology_bands:
            value = sampled.get(band)
            feature_data[band] = value
        
        return ee.Feature(point.geometry(), feature_data)
    
    # Create a grid of points within the AOI
    bounds = aoi.bounds()
    min_lon = bounds.getInfo()['coordinates'][0][0][0]
    max_lon = bounds.getInfo()['coordinates'][0][2][0]
    min_lat = bounds.getInfo()['coordinates'][0][0][1]
    max_lat = bounds.getInfo()['coordinates'][0][2][1]
    
    # Create point grid
    grid_size = 0.1  # degrees
    lons = ee.List.sequence(min_lon, max_lon, grid_size)
    lats = ee.List.sequence(min_lat, max_lat, grid_size)
    
    def create_point(lon, lat):
        return ee.Feature(ee.Geometry.Point([lon, lat]))
    
    points = lons.map(lambda lon: lats.map(lambda lat: create_point(lon, lat))).flatten()
    point_collection = ee.FeatureCollection(points)
    
    # Sample phenology data
    sampled_collection = point_collection.map(sample_phenology)
    
    # Convert to pandas DataFrame
    try:
        # Get the data
        data = sampled_collection.getInfo()
        
        if 'features' not in data:
            logger.warning("No features found in MCD12Q2 data")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for feature in data['features']:
            props = feature['properties']
            records.append(props)
        
        df = pd.DataFrame(records)
        
        if len(df) == 0:
            logger.warning("No valid phenology data found")
            return pd.DataFrame()
        
        logger.info(f"Fetched {len(df)} phenology records")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch MCD12Q2 data: {e}")
        return pd.DataFrame()


def extract_bloom_onsets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract bloom onset dates from MCD12Q2 phenology data.
    
    Args:
        df: DataFrame with MCD12Q2 phenology data
        
    Returns:
        DataFrame with bloom onset labels
    """
    logger.info("Extracting bloom onset dates...")
    
    if len(df) == 0:
        return pd.DataFrame()
    
    labels = []
    
    for idx, row in df.iterrows():
        # Get coordinates
        lat = row['latitude']
        lon = row['longitude']
        location_id = f"{lat:.4f}_{lon:.4f}"
        
        # Look for the first valid greenup date
        greenup_dates = []
        for i in range(1, 5):  # Check up to 4 cycles
            greenup_col = f'Greenup_{i}'
            if greenup_col in row and pd.notna(row[greenup_col]):
                # MCD12Q2 dates are in day of year
                day_of_year = int(row[greenup_col])
                if 1 <= day_of_year <= 366:  # Valid day of year
                    greenup_dates.append(day_of_year)
        
        if not greenup_dates:
            continue
        
        # Use the earliest greenup date as bloom onset
        onset_doy = min(greenup_dates)
        
        # Convert day of year to date (assuming 2020 as reference year)
        try:
            onset_date = pd.to_datetime(f'2020-{onset_doy:03d}', format='%Y-%j')
            onset_date_str = onset_date.strftime('%Y-%m-%d')
        except:
            continue
        
        # Calculate quality score based on data availability
        quality_score = len(greenup_dates) / 4.0  # Normalize to 0-1
        
        # Get additional phenology metrics
        peak_dates = []
        for i in range(1, 5):
            peak_col = f'Peak_{i}'
            if peak_col in row and pd.notna(row[peak_col]):
                peak_doy = int(row[peak_col])
                if 1 <= peak_doy <= 366:
                    peak_dates.append(peak_doy)
        
        peak_doy = min(peak_dates) if peak_dates else None
        
        label = {
            'location_id': location_id,
            'year': 2020,  # MCD12Q2 is annual
            'onset_date': onset_date_str,
            'onset_doy': onset_doy,
            'peak_doy': peak_doy,
            'label_quality_score': quality_score,
            'latitude': lat,
            'longitude': lon,
            'data_source': 'MCD12Q2'
        }
        
        labels.append(label)
    
    if not labels:
        logger.warning("No valid bloom onsets found in MCD12Q2 data")
        return pd.DataFrame()
    
    labels_df = pd.DataFrame(labels)
    logger.info(f"Extracted {len(labels_df)} bloom onset labels")
    
    return labels_df


def main():
    """Main function for MCD12Q2 labeling."""
    parser = argparse.ArgumentParser(description='Create bloom labels from MCD12Q2 phenology data')
    parser.add_argument('--aoi', required=True, 
                       help='Area of interest: GeoJSON file path or bbox "x1,y1,x2,y2"')
    parser.add_argument('--start', required=True, 
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', required=True, 
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--output', 
                       help='Output labels CSV file path (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Parse AOI
    if args.aoi.endswith('.geojson') or args.aoi.endswith('.json'):
        import json
        with open(args.aoi, 'r') as f:
            geojson = json.load(f)
        aoi = ee.Geometry(geojson)
    else:
        coords = [float(x.strip()) for x in args.aoi.split(',')]
        if len(coords) != 4:
            logger.error("Bbox must have 4 coordinates: x1,y1,x2,y2")
            sys.exit(1)
        x1, y1, x2, y2 = coords
        aoi = ee.Geometry.Rectangle([x1, y1, x2, y2])
    
    # Set up output path
    if args.output:
        output_path = Path(args.output)
    else:
        aoi_str = args.aoi.replace(',', '_').replace(' ', '_')
        output_path = DATA_PROCESSED_DIR / f"mcd12q2_{aoi_str}_{args.start}_{args.end}_labels.csv"
    
    # Fetch MCD12Q2 data
    phenology_df = fetch_mcd12q2_data(aoi, args.start, args.end)
    
    if len(phenology_df) == 0:
        logger.error("No MCD12Q2 data available. Consider using change-point detection instead.")
        sys.exit(1)
    
    # Extract bloom onsets
    labels_df = extract_bloom_onsets(phenology_df)
    
    if len(labels_df) == 0:
        logger.error("No bloom onsets found in MCD12Q2 data")
        sys.exit(1)
    
    # Save labels
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(output_path, index=False)
        logger.info(f"Labels saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save labels: {e}")
        sys.exit(1)
    
    logger.info("MCD12Q2 labeling completed!")


if __name__ == '__main__':
    main()

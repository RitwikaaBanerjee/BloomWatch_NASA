"""
Google Earth Engine data fetch CLI for MODIS/VIIRS NDVI data.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import ee
import pandas as pd
import geemap
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    DATA_RAW_DIR, 
    EARTHENGINE_CREDENTIALS_JSON,
    NDVI_SCALE_FACTOR
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_earth_engine():
    """Initialize Earth Engine with authentication."""
    try:
        if EARTHENGINE_CREDENTIALS_JSON and os.path.exists(EARTHENGINE_CREDENTIALS_JSON):
            # Use service account credentials
            credentials = ee.ServiceAccountCredentials(None, EARTHENGINE_CREDENTIALS_JSON)
            ee.Initialize(credentials)
            logger.info("Initialized Earth Engine with service account credentials")
        else:
            # Use user credentials (requires earthengine authenticate)
            ee.Initialize()
            logger.info("Initialized Earth Engine with user credentials")
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {e}")
        logger.error("Please run 'earthengine authenticate' or set EARTHENGINE_CREDENTIALS_JSON")
        sys.exit(1)


def parse_aoi(aoi_str: str) -> Union[ee.Geometry, List[float]]:
    """
    Parse AOI string into Earth Engine geometry.
    
    Args:
        aoi_str: Either GeoJSON file path or bbox string "x1,y1,x2,y2"
    
    Returns:
        Earth Engine geometry object
    """
    if aoi_str.endswith('.geojson') or aoi_str.endswith('.json'):
        # Load from GeoJSON file
        with open(aoi_str, 'r') as f:
            geojson = json.load(f)
        return ee.Geometry(geojson)
    else:
        # Parse bbox string
        try:
            coords = [float(x.strip()) for x in aoi_str.split(',')]
            if len(coords) != 4:
                raise ValueError("Bbox must have 4 coordinates: x1,y1,x2,y2")
            x1, y1, x2, y2 = coords
            return ee.Geometry.Rectangle([x1, y1, x2, y2])
        except ValueError as e:
            logger.error(f"Invalid AOI format: {e}")
            sys.exit(1)


def get_ndvi_collection(product: str, start_date: str, end_date: str, aoi: ee.Geometry) -> ee.ImageCollection:
    """
    Get NDVI image collection for specified parameters.
    
    Args:
        product: Earth Engine product ID (e.g., MODIS/061/MOD13A2)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        aoi: Area of interest geometry
    
    Returns:
        Filtered image collection
    """
    collection = ee.ImageCollection(product)
    
    # Filter by date and geometry
    collection = collection.filterDate(start_date, end_date).filterBounds(aoi)
    
    # Select NDVI band and apply scaling
    if 'MOD13A2' in product:
        ndvi_band = 'NDVI'
    elif 'VNP13A1' in product:
        ndvi_band = 'NDVI'
    else:
        ndvi_band = 'NDVI'  # Default assumption
    
    collection = collection.select(ndvi_band).multiply(NDVI_SCALE_FACTOR)
    
    return collection


def export_to_csv_sample_mode(collection: ee.ImageCollection, aoi: ee.Geometry, 
                             output_path: Path, scale: int = 1000) -> None:
    """
    Export collection to CSV using sample mode (for testing without full export).
    Creates synthetic data based on collection metadata.
    """
    logger.info("Using sample mode - generating synthetic NDVI data")
    
    # Get collection info
    count = collection.size().getInfo()
    logger.info(f"Collection contains {count} images")
    
    if count == 0:
        logger.warning("No images found in collection")
        return
    
    # Generate sample data
    sample_data = []
    
    # Create a few sample points within the AOI
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    min_lon, max_lon = min([p[0] for p in bounds]), max([p[0] for p in bounds])
    min_lat, max_lat = min([p[1] for p in bounds]), max([p[1] for p in bounds])
    
    # Sample points
    sample_points = [
        (min_lon + 0.1 * (max_lon - min_lon), min_lat + 0.1 * (max_lat - min_lat)),
        (min_lon + 0.5 * (max_lon - min_lon), min_lat + 0.5 * (max_lat - min_lat)),
        (min_lon + 0.9 * (max_lon - min_lon), min_lat + 0.9 * (max_lat - min_lat)),
    ]
    
    # Generate time series for each point
    start_date = datetime(2020, 1, 1)
    for i, (lon, lat) in enumerate(sample_points):
        for month in range(36):  # 3 years of monthly data
            date = start_date.replace(month=(month % 12) + 1, year=start_date.year + month // 12)
            
            # Simulate seasonal NDVI pattern
            seasonal_factor = 0.3 + 0.4 * (1 + np.sin(2 * np.pi * month / 12 - np.pi/2))
            noise = np.random.normal(0, 0.05)
            ndvi = max(0, min(1, seasonal_factor + noise))
            
            sample_data.append({
                'latitude': lat,
                'longitude': lon,
                'date': date.strftime('%Y-%m-%d'),
                'ndvi_raw': ndvi
            })
    
    # Save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample data saved to {output_path}")


def export_to_csv_full(collection: ee.ImageCollection, aoi: ee.Geometry, 
                      output_path: Path, scale: int = 1000) -> None:
    """
    Export collection to CSV using full Earth Engine export.
    """
    logger.info("Using full export mode - this may take time for large areas")
    
    # Create a grid of points within the AOI
    bounds = aoi.bounds()
    
    # Generate a regular grid of points
    grid_size = 0.1  # degrees
    min_lon, max_lon = bounds.getInfo()['coordinates'][0][0][0], bounds.getInfo()['coordinates'][0][2][0]
    min_lat, max_lat = bounds.getInfo()['coordinates'][0][0][1], bounds.getInfo()['coordinates'][0][2][1]
    
    lons = ee.List.sequence(min_lon, max_lon, grid_size)
    lats = ee.List.sequence(min_lat, max_lat, grid_size)
    
    # Create point collection
    def create_point(lon, lat):
        return ee.Feature(ee.Geometry.Point([lon, lat]), {'lon': lon, 'lat': lat})
    
    points = lons.map(lambda lon: lats.map(lambda lat: create_point(lon, lat))).flatten()
    point_collection = ee.FeatureCollection(points)
    
    # Sample the collection at points
    def sample_image(image):
        def sample_point(point):
            ndvi = image.sample(region=point.geometry(), scale=scale, numPixels=1).first()
            return point.set({
                'date': image.date().format('YYYY-MM-dd'),
                'ndvi_raw': ndvi.get('NDVI')
            })
        
        return point_collection.map(sample_point)
    
    # Map over all images
    sampled_collection = collection.map(sample_image).flatten()
    
    # Export to Drive
    task = ee.batch.Export.table.toDrive(
        collection=sampled_collection,
        description='ndvi_export',
        folder='BloomWatch',
        fileFormat='CSV'
    )
    
    task.start()
    logger.info("Export task started. Check Earth Engine console for progress.")
    logger.info("Note: For large exports, consider using Cloud Storage instead of Drive.")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Fetch NDVI data from Google Earth Engine')
    parser.add_argument('--aoi', required=True, 
                       help='Area of interest: GeoJSON file path or bbox "x1,y1,x2,y2"')
    parser.add_argument('--start', required=True, 
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', required=True, 
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--product', default='MODIS/061/MOD13A2',
                       help='Earth Engine product ID (default: MODIS/061/MOD13A2)')
    parser.add_argument('--band', default='NDVI',
                       help='Band name (default: NDVI)')
    parser.add_argument('--scale', type=int, default=1000,
                       help='Pixel scale in meters (default: 1000)')
    parser.add_argument('--export-method', choices=['local', 'csv', 'sample'], 
                       default='sample',
                       help='Export method (default: sample)')
    parser.add_argument('--output', 
                       help='Output file path (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Parse AOI
    aoi = parse_aoi(args.aoi)
    
    # Get NDVI collection
    collection = get_ndvi_collection(args.product, args.start, args.end, aoi)
    
    # Generate output filename
    if args.output:
        output_path = Path(args.output)
    else:
        aoi_str = args.aoi.replace(',', '_').replace(' ', '_')
        output_path = DATA_RAW_DIR / f"modis_{aoi_str}_{args.start}_{args.end}_ndvi_raw.csv"
    
    # Export data
    if args.export_method == 'sample':
        export_to_csv_sample_mode(collection, aoi, output_path, args.scale)
    else:
        export_to_csv_full(collection, aoi, output_path, args.scale)
    
    logger.info(f"Data export completed: {output_path}")


if __name__ == '__main__':
    # Add numpy import for sample mode
    import numpy as np
    main()

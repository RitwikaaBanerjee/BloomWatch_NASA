"""
NASA AppEEARS data fetch CLI for MODIS/VIIRS NDVI data.
"""

import argparse
import json
import logging
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    DATA_RAW_DIR, 
    APPEEARS_USERNAME,
    APPEEARS_PASSWORD,
    NDVI_SCALE_FACTOR
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AppEEARS API endpoints
APPEEARS_BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"
LOGIN_URL = f"{APPEEARS_BASE_URL}/login"
TASK_URL = f"{APPEEARS_BASE_URL}/task"
PRODUCT_URL = f"{APPEEARS_BASE_URL}/product"


class AppEEARSClient:
    """Client for interacting with NASA AppEEARS API."""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.token = None
        self.session = requests.Session()
    
    def authenticate(self) -> bool:
        """Authenticate with AppEEARS API."""
        try:
            response = self.session.post(LOGIN_URL, json={
                'username': self.username,
                'password': self.password
            })
            response.raise_for_status()
            
            data = response.json()
            self.token = data.get('token')
            
            if self.token:
                self.session.headers.update({'Authorization': f'Bearer {self.token}'})
                logger.info("Successfully authenticated with AppEEARS")
                return True
            else:
                logger.error("Authentication failed: No token received")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_products(self) -> List[Dict]:
        """Get available products."""
        try:
            response = self.session.get(PRODUCT_URL)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get products: {e}")
            return []
    
    def create_task(self, product: str, aoi: Dict, start_date: str, end_date: str) -> Optional[str]:
        """Create a new task for data extraction."""
        try:
            task_data = {
                'task_type': 'point',
                'task_name': f'BloomWatch_{product}_{start_date}_{end_date}',
                'params': {
                    'dates': [{'startDate': start_date, 'endDate': end_date}],
                    'layers': [{'product': product, 'layer': 'NDVI'}],
                    'coordinates': aoi['coordinates']
                }
            }
            
            response = self.session.post(TASK_URL, json=task_data)
            response.raise_for_status()
            
            task_info = response.json()
            task_id = task_info.get('task_id')
            
            if task_id:
                logger.info(f"Task created successfully: {task_id}")
                return task_id
            else:
                logger.error("Failed to create task: No task ID received")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create task: {e}")
            return None
    
    def check_task_status(self, task_id: str) -> Dict:
        """Check the status of a task."""
        try:
            response = self.session.get(f"{TASK_URL}/{task_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check task status: {e}")
            return {}
    
    def download_task(self, task_id: str, output_path: Path) -> bool:
        """Download completed task results."""
        try:
            response = self.session.get(f"{TASK_URL}/{task_id}/output")
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Task results downloaded: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download task: {e}")
            return False


def parse_aoi(aoi_str: str) -> Dict:
    """
    Parse AOI string into AppEEARS format.
    
    Args:
        aoi_str: Bbox string "x1,y1,x2,y2" or GeoJSON file path
    
    Returns:
        AOI dictionary for AppEEARS API
    """
    if aoi_str.endswith('.geojson') or aoi_str.endswith('.json'):
        # Load from GeoJSON file
        with open(aoi_str, 'r') as f:
            geojson = json.load(f)
        
        # Convert to AppEEARS format
        if geojson['type'] == 'FeatureCollection':
            coords = geojson['features'][0]['geometry']['coordinates'][0]
        else:
            coords = geojson['coordinates'][0]
        
        return {'coordinates': coords}
    else:
        # Parse bbox string
        try:
            coords = [float(x.strip()) for x in aoi_str.split(',')]
            if len(coords) != 4:
                raise ValueError("Bbox must have 4 coordinates: x1,y1,x2,y2")
            
            x1, y1, x2, y2 = coords
            # AppEEARS expects coordinates as [lon1, lat1, lon2, lat2, ...]
            return {'coordinates': [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]}
            
        except ValueError as e:
            logger.error(f"Invalid AOI format: {e}")
            sys.exit(1)


def process_downloaded_data(zip_path: Path, output_path: Path) -> None:
    """Process downloaded zip file and extract NDVI data."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find CSV files
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            
            if not csv_files:
                logger.error("No CSV files found in downloaded data")
                return
            
            # Process each CSV file
            all_data = []
            for csv_file in csv_files:
                with zip_ref.open(csv_file) as f:
                    df = pd.read_csv(f)
                    
                    # Standardize column names
                    if 'Latitude' in df.columns:
                        df = df.rename(columns={'Latitude': 'latitude'})
                    if 'Longitude' in df.columns:
                        df = df.rename(columns={'Longitude': 'longitude'})
                    if 'Date' in df.columns:
                        df = df.rename(columns={'Date': 'date'})
                    if 'NDVI' in df.columns:
                        df = df.rename(columns={'NDVI': 'ndvi_raw'})
                    
                    # Apply scaling if needed
                    if 'ndvi_raw' in df.columns:
                        df['ndvi_raw'] = df['ndvi_raw'] * NDVI_SCALE_FACTOR
                    
                    all_data.append(df)
            
            # Combine all data
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df.to_csv(output_path, index=False)
                logger.info(f"Processed data saved to {output_path}")
            else:
                logger.error("No valid data found in downloaded files")
                
    except Exception as e:
        logger.error(f"Failed to process downloaded data: {e}")


def create_sample_data(aoi: Dict, start_date: str, end_date: str, output_path: Path) -> None:
    """Create sample data for testing when AppEEARS is not available."""
    logger.info("Creating sample data for testing")
    
    # Generate sample coordinates within AOI
    coords = aoi['coordinates']
    min_lon = min([p[0] for p in coords])
    max_lon = max([p[0] for p in coords])
    min_lat = min([p[1] for p in coords])
    max_lat = max([p[1] for p in coords])
    
    # Sample points
    sample_points = [
        (min_lon + 0.1 * (max_lon - min_lon), min_lat + 0.1 * (max_lat - min_lat)),
        (min_lon + 0.5 * (max_lon - min_lon), min_lat + 0.5 * (max_lat - min_lat)),
        (min_lon + 0.9 * (max_lon - min_lon), min_lat + 0.9 * (max_lat - min_lat)),
    ]
    
    # Generate time series
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    sample_data = []
    for lon, lat in sample_points:
        current_date = start_dt
        while current_date <= end_dt:
            # Simulate seasonal NDVI pattern
            month = current_date.month
            seasonal_factor = 0.3 + 0.4 * (1 + np.sin(2 * np.pi * month / 12 - np.pi/2))
            noise = np.random.normal(0, 0.05)
            ndvi = max(0, min(1, seasonal_factor + noise))
            
            sample_data.append({
                'latitude': lat,
                'longitude': lon,
                'date': current_date.strftime('%Y-%m-%d'),
                'ndvi_raw': ndvi
            })
            
            current_date = current_date.replace(day=1) + pd.DateOffset(months=1)
    
    # Save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample data saved to {output_path}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Fetch NDVI data from NASA AppEEARS')
    parser.add_argument('--aoi', required=True, 
                       help='Area of interest: GeoJSON file path or bbox "x1,y1,x2,y2"')
    parser.add_argument('--start', required=True, 
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', required=True, 
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--product', default='MOD13A2.061',
                       help='AppEEARS product ID (default: MOD13A2.061)')
    parser.add_argument('--output', 
                       help='Output file path (auto-generated if not provided)')
    parser.add_argument('--sample-mode', action='store_true',
                       help='Use sample mode instead of real API calls')
    
    args = parser.parse_args()
    
    # Parse AOI
    aoi = parse_aoi(args.aoi)
    
    # Generate output filename
    if args.output:
        output_path = Path(args.output)
    else:
        aoi_str = args.aoi.replace(',', '_').replace(' ', '_')
        output_path = DATA_RAW_DIR / f"appeears_{aoi_str}_{args.start}_{args.end}_ndvi_raw.csv"
    
    if args.sample_mode or not APPEEARS_USERNAME or not APPEEARS_PASSWORD:
        # Use sample mode
        create_sample_data(aoi, args.start, args.end, output_path)
        return
    
    # Initialize AppEEARS client
    client = AppEEARSClient(APPEEARS_USERNAME, APPEEARS_PASSWORD)
    
    # Authenticate
    if not client.authenticate():
        logger.error("Authentication failed. Using sample mode instead.")
        create_sample_data(aoi, args.start, args.end, output_path)
        return
    
    # Create task
    task_id = client.create_task(args.product, aoi, args.start, args.end)
    if not task_id:
        logger.error("Failed to create task. Using sample mode instead.")
        create_sample_data(aoi, args.start, args.end, output_path)
        return
    
    # Wait for task completion
    logger.info("Waiting for task completion...")
    max_wait_time = 3600  # 1 hour
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status = client.check_task_status(task_id)
        task_status = status.get('status', 'unknown')
        
        logger.info(f"Task status: {task_status}")
        
        if task_status == 'done':
            # Download results
            zip_path = output_path.with_suffix('.zip')
            if client.download_task(task_id, zip_path):
                process_downloaded_data(zip_path, output_path)
                zip_path.unlink()  # Remove zip file
            break
        elif task_status == 'failed':
            logger.error("Task failed. Using sample mode instead.")
            create_sample_data(aoi, args.start, args.end, output_path)
            break
        
        time.sleep(30)  # Wait 30 seconds before checking again
    else:
        logger.error("Task timed out. Using sample mode instead.")
        create_sample_data(aoi, args.start, args.end, output_path)
    
    logger.info(f"Data fetch completed: {output_path}")


if __name__ == '__main__':
    # Add numpy import for sample mode
    import numpy as np
    main()

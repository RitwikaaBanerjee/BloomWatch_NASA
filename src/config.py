"""
Configuration module for BloomWatch pipeline.
Reads environment variables and sets default paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'data'))
DATA_RAW_DIR = Path(os.getenv('DATA_RAW_DIR', OUTPUT_DIR / 'raw'))
DATA_PROCESSED_DIR = Path(os.getenv('DATA_PROCESSED_DIR', OUTPUT_DIR / 'processed'))
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'models'))
REPORTS_DIR = Path(os.getenv('REPORTS_DIR', 'reports'))

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API settings
API_HOST = os.getenv('API_HOST', '127.0.0.1')
API_PORT = int(os.getenv('API_PORT', '8001'))

# Model settings
DEFAULT_MODEL_TYPE = os.getenv('DEFAULT_MODEL_TYPE', 'randomforest')
DEFAULT_TASK = os.getenv('DEFAULT_TASK', 'classification')

# Authentication credentials
EARTHENGINE_CREDENTIALS_JSON = os.getenv('EARTHENGINE_CREDENTIALS_JSON')
APPEEARS_USERNAME = os.getenv('APPEEARS_USERNAME')
APPEEARS_PASSWORD = os.getenv('APPEEARS_PASSWORD')
EARTHDATA_USERNAME = os.getenv('EARTHDATA_USERNAME')
EARTHDATA_PASSWORD = os.getenv('EARTHDATA_PASSWORD')

# Data processing settings
NDVI_SCALE_FACTOR = 0.0001  # MODIS NDVI scaling factor
DEFAULT_FREQUENCY = 'M'  # Monthly frequency
SMOOTHING_WINDOW = 5  # Savitzky-Golay window size
SMOOTHING_POLYORDER = 2  # Savitzky-Golay polynomial order

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

XGBOOST_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

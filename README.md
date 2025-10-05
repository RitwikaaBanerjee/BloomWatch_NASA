# BloomWatch

A reproducible pipeline for downloading NASA satellite vegetation data (MODIS/VIIRS), preprocessing NDVI time-series, generating bloom labels, training ML models to detect/predict bloom timing, and exposing an API for predictions with a minimal demo frontend.

## Overview

BloomWatch provides an end-to-end solution for vegetation bloom detection and prediction using satellite data. The pipeline includes:

- **Data Ingestion**: NASA AppEEARS for satellite data
- **Preprocessing**: NDVI time-series cleaning, smoothing, and resampling
- **Labeling**: Automatic bloom onset detection using change-point analysis
- **ML Pipeline**: Feature engineering, model training, and evaluation
- **API**: FastAPI endpoints for real-time predictions
- **Demo**: Streamlit interface for interactive exploration

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- NASA AppEEARS account (for satellite data)

## Setup

1. **Clone and setup environment:**
```bash
git clone https://github.com/RitwikaaBanerjee/BloomWatch_NASA
cd NASA
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure credentials:**
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Set up NASA AppEEARS credentials:**
```bash
# Set APPEEARS_USERNAME and APPEEARS_PASSWORD in .env
```

## Quick Start (Sample Mode)

Run the demo with synthetic data (no credentials required):

```bash
# Start the demo
streamlit run src/demo/streamlit_app.py
```

## Full Pipeline (Real Data)

### 1. Fetch Data

**NASA AppEEARS:**
```bash
python src/data_fetch/fetch_appeears.py --aoi "68,6,97,37" --start 2019-01-01 --end 2023-12-31
```

### 2. Preprocess Data
```bash
python src/preprocessing/preprocess_ndvi.py --input data/raw/modis_68,6,97,37_2019-01-01_2023-12-31_ndvi_raw.csv --output data/processed/india_monthly_ndvi.csv
```

### 3. Generate Labels
```bash
python src/labeling/label_from_change_point.py --input data/processed/india_monthly_ndvi.csv --output data/processed/india_labels.csv
```

### 4. Create Features
```bash
python src/features/features.py --input data/processed/india_monthly_ndvi.csv --output data/processed/india_features.csv
```

### 5. Train Model
```bash
python src/models/train_model.py --task classification --model randomforest
```

### 6. Start API
```bash
uvicorn src.api.app:app --reload
```

### 7. Run Demo
```bash
streamlit run src/demo/streamlit_app.py
```

## Authentication

### NASA AppEEARS
1. Create account at [appeears.earthdatacloud.nasa.gov](https://appeears.earthdatacloud.nasa.gov)
2. Set `APPEEARS_USERNAME` and `APPEEARS_PASSWORD` in `.env`

## Output Structure

- `data/raw/` - Raw satellite data CSV files
- `data/processed/` - Preprocessed NDVI, labels, and features
- `models/` - Trained ML models
- `reports/` - Evaluation metrics and figures

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Get bloom predictions for coordinates
- `GET /model_info` - Model metadata and performance

## Testing

```bash
pytest tests/
```

## Contributors

- **@ritwikaabanerjee** nasa data API , preprocessing and evaluation  
- **@prahants**  labeling, feature engineering, and model training    




## License

MIT License

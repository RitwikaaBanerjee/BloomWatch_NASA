# BloomWatch Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_setup.py
```

### 3. Run the Demo (Sample Mode)
```bash
streamlit run src/demo/streamlit_app.py
```

### 4. Start the API
```bash
uvicorn src.api.app:app --reload
```

## 📊 Sample Data Pipeline

The system comes with sample data for immediate testing:

### Data Fetching (Sample Mode)
```bash
# GEE sample mode
python src/data_fetch/fetch_gee.py --aoi "40.7,-74.0,40.8,-73.9" --start 2020-01-01 --end 2022-12-31 --export-method sample

# AppEEARS sample mode  
python src/data_fetch/fetch_appeears.py --aoi "40.7,-74.0,40.8,-73.9" --start 2020-01-01 --end 2022-12-31 --sample-mode
```

### Preprocessing
```bash
python src/preprocessing/preprocess_ndvi.py --input data/raw/sample_ndvi_raw.csv --output data/processed/sample_processed.csv
```

### Feature Engineering
```bash
python src/features/features.py --input data/processed/sample_processed.csv --output data/processed/sample_features.csv
```

### Model Training
```bash
python src/models/train_model.py --task classification --model randomforest
```

### Label Generation
```bash
python src/labeling/label_from_change_point.py --input data/processed/sample_processed.csv --output data/processed/sample_labels.csv
```

## 🌍 Real Data Pipeline (Requires Credentials)

### 1. Set up Environment
```bash
cp env.example .env
# Edit .env with your credentials
```

### 2. Authenticate with Earth Engine
```bash
earthengine authenticate
```

### 3. Fetch Real Data
```bash
python src/data_fetch/fetch_gee.py --aoi "40.7,-74.0,40.8,-73.9" --start 2020-01-01 --end 2022-12-31
```

## 🔧 Available Commands

### Data Fetching
- `python src/data_fetch/fetch_gee.py` - Google Earth Engine data
- `python src/data_fetch/fetch_appeears.py` - NASA AppEEARS data

### Preprocessing
- `python src/preprocessing/preprocess_ndvi.py` - Clean and smooth NDVI data
- `python src/labeling/label_from_change_point.py` - Generate bloom labels

### Machine Learning
- `python src/features/features.py` - Create ML features
- `python src/models/train_model.py` - Train models
- `python src/models/evaluate.py` - Evaluate models

### Applications
- `streamlit run src/demo/streamlit_app.py` - Interactive demo
- `uvicorn src.api.app:app --reload` - API server

## 📁 Project Structure

```
NASA/
├── src/                    # Source code
│   ├── data_fetch/        # Data ingestion
│   ├── preprocessing/     # Data cleaning
│   ├── labeling/         # Bloom detection
│   ├── features/         # Feature engineering
│   ├── models/           # ML training
│   ├── api/              # FastAPI server
│   └── demo/             # Streamlit demo
├── data/                  # Data storage
│   ├── raw/              # Raw satellite data
│   └── processed/        # Processed data
├── models/               # Trained models
├── reports/              # Results and plots
├── tests/                # Unit tests
└── requirements.txt      # Dependencies
```

## 🎯 Key Features

- **Multiple Data Sources**: Google Earth Engine + NASA AppEEARS
- **Robust Preprocessing**: Cloud masking, smoothing, interpolation
- **Change-Point Detection**: Automatic bloom onset detection
- **ML Pipeline**: Random Forest + XGBoost models
- **Real-time API**: FastAPI endpoints for predictions
- **Interactive Demo**: Streamlit web interface
- **Sample Mode**: Works without credentials for testing

## 🔍 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Bloom predictions
- `GET /model_info` - Model metadata
- `GET /download_csv` - Download results

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_preprocess.py -v
```

## 📚 Next Steps

1. **Explore the Demo**: Run the Streamlit app to see the interface
2. **Try the API**: Start the API server and test endpoints
3. **Add Real Data**: Set up credentials and fetch real satellite data
4. **Customize Models**: Modify hyperparameters and add new features
5. **Deploy**: Use Docker or cloud platforms for production

## 🆘 Troubleshooting

- **Import Errors**: Run `pip install -r requirements.txt`
- **Earth Engine Issues**: Run `earthengine authenticate`
- **API Not Starting**: Check if port 8000 is available
- **Demo Not Loading**: Ensure all dependencies are installed

## 📖 Documentation

- See `README.md` for detailed documentation
- Check individual module docstrings for API details
- Run `python -m src.module_name --help` for CLI help

.PHONY: help install test clean setup-data run-api run-demo

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	pytest tests/ -v

clean: ## Clean up generated files
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.csv
	rm -rf models/*.joblib
	rm -rf reports/*.json
	rm -rf reports/figures/*.png

setup-data: ## Create sample data for testing
	python -c "from src.demo.sample_data import create_sample_data; create_sample_data()"

run-api: ## Start the FastAPI server
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

run-demo: ## Start the Streamlit demo
	streamlit run src/demo/streamlit_app.py

fetch-sample: ## Fetch sample data using GEE
	python src/data_fetch/fetch_gee.py --aoi "68,6,97,37" --start 2020-01-01 --end 2022-12-31 --export-method sample

preprocess-sample: ## Preprocess sample data
	python src/preprocessing/preprocess_ndvi.py --input data/raw/sample_ndvi_raw.csv --output data/processed/sample_monthly_ndvi.csv

train-sample: ## Train model on sample data
	python src/models/train_model.py --task classification --model randomforest --input data/processed/sample_features.csv

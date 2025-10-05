# NASA AppEEARS Integration - Changes Summary

## Overview
Updated BloomWatch to use **NASA AppEEARS** instead of Google Earth Engine for fetching satellite NDVI data.

## Changes Made

### 1. API Backend (`src/api/app.py`)
- **Removed**: Google Earth Engine (`ee`) dependency
- **Added**: NASA AppEEARS client integration
- **Updated**: `fetch_ndvi_data()` function to use AppEEARS API
- **Updated**: Product field default from `"MODIS/061/MOD13A2"` to `"MOD13A2.061"` (AppEEARS format)
- **Fixed**: Sample data generator now uses coordinate-based RNG seed so results vary by location/time

### 2. Streamlit Demo (`src/demo/streamlit_app.py`)
- **Updated**: Satellite product dropdown to use AppEEARS format:
  - `MOD13A2.061` (MODIS Terra 16-day 1km)
  - `MOD13Q1.061` (MODIS Terra 16-day 250m)
  - `VNP13A1.001` (VIIRS 16-day 500m)
- **Fixed**: Sample data generator (`generate_sample_data()`) now varies by latitude/longitude
- **Updated**: About section to reflect AppEEARS as data source

### 3. Configuration
AppEEARS credentials are already configured in `env.example`:
```env
APPEEARS_USERNAME=prahants
APPEEARS_PASSWORD=392004.Nasa!
```

## How It Works Now

### Real Data Mode (with AppEEARS credentials)
1. API receives prediction request with lat/lon and date range
2. Attempts to authenticate with NASA AppEEARS
3. Creates a task to extract NDVI data for the location
4. **Note**: AppEEARS tasks take minutes to hours to complete
5. For real-time API responses, falls back to sample data immediately

### Sample Data Mode (default for real-time)
1. Generates synthetic NDVI data based on:
   - **Latitude**: Affects amplitude (lower near poles)
   - **Longitude**: Adds phase shift to seasonal pattern
   - **Date range**: Determines number of data points
   - **Coordinate-based seed**: Ensures results vary by location but are stable per input

## Why Sample Mode is Default

NASA AppEEARS is designed for **batch processing**, not real-time queries:
- Tasks typically take **5-60 minutes** to complete
- Results are delivered as downloadable files
- Not suitable for interactive web applications

## Recommendations

### For Real-Time Predictions
Continue using the **coordinate-aware sample data** which now properly varies by location.

### For Training Data
Use the CLI tool to fetch real data in batch:
```bash
python -m src.data_fetch.fetch_appeears \
  --aoi "lon,lat,lon,lat" \
  --start "2020-01-01" \
  --end "2023-12-31" \
  --product "MOD13A2.061"
```

### For Production
Consider implementing:
1. **Background job queue** for AppEEARS tasks
2. **Caching layer** for previously fetched locations
3. **Pre-computed predictions** for common regions

## Testing the Fix

1. **Restart the API server**:
   ```bash
   python -m src.api.app
   ```

2. **Restart Streamlit**:
   ```bash
   streamlit run src/demo/streamlit_app.py
   ```

3. **Test coordinate variation**:
   - Try different lat/lon values (e.g., 40.7128, -74.0060 vs 34.0522, -118.2437)
   - Click "Analyze Bloom"
   - Verify "Bloom Events" and plots change with different coordinates

## Files Modified
- `src/api/app.py` - Switched to AppEEARS, fixed sample data
- `src/demo/streamlit_app.py` - Updated UI, fixed sample data generator
- `APPEEARS_INTEGRATION.md` - This documentation

## Next Steps
1. Ensure `.env` file has your AppEEARS credentials (copy from `env.example`)
2. Test the application with different coordinates
3. For production, implement async task handling for real AppEEARS data

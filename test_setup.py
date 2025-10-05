"""
Test script to verify BloomWatch setup and basic functionality.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import src.config
        print("✅ Config module imported")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        import src.preprocessing.preprocess_ndvi
        print("✅ Preprocessing module imported")
    except Exception as e:
        print(f"❌ Preprocessing import failed: {e}")
        return False
    
    try:
        import src.features.features
        print("✅ Features module imported")
    except Exception as e:
        print(f"❌ Features import failed: {e}")
        return False
    
    try:
        import src.models.train_model
        print("✅ Models module imported")
    except Exception as e:
        print(f"❌ Models import failed: {e}")
        return False
    
    try:
        import src.api.app
        print("✅ API module imported")
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False
    
    try:
        import src.demo.streamlit_app
        print("✅ Demo module imported")
    except Exception as e:
        print(f"❌ Demo import failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR
        print(f"✅ Data directories: {DATA_RAW_DIR}, {DATA_PROCESSED_DIR}, {MODEL_DIR}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_sample_data():
    """Test sample data creation."""
    print("\nTesting sample data creation...")
    
    try:
        from src.demo.sample_data import create_sample_data
        create_sample_data()
        print("✅ Sample data created successfully")
        return True
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\nTesting preprocessing...")
    
    try:
        from src.demo.sample_data import create_sample_ndvi_data
        from src.preprocessing.preprocess_ndvi import clean_data, apply_smoothing
        
        # Create sample data
        ndvi_df = create_sample_ndvi_data(n_locations=2, n_years=1)
        
        # Test cleaning
        cleaned_df = clean_data(ndvi_df)
        assert len(cleaned_df) > 0
        print("✅ Data cleaning works")
        
        # Test smoothing
        smoothed_df = apply_smoothing(cleaned_df)
        assert 'ndvi_smoothed' in smoothed_df.columns
        print("✅ Data smoothing works")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False


def test_model_training():
    """Test model training."""
    print("\nTesting model training...")
    
    try:
        from src.models.train_model import create_sample_data
        create_sample_data()
        print("✅ Sample model training works")
        return True
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("BloomWatch Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_sample_data,
        test_preprocessing,
        test_model_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! BloomWatch is ready to use.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

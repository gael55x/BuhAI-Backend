#!/usr/bin/env python3
"""
Test script for LSTM integration in BuhAI-Backend.

This script tests the LSTM prediction functionality to ensure everything
is working correctly after integration.
"""

import sys
import os
from pathlib import Path

# Add model directory to Python path
sys.path.append(str(Path(__file__).parent / "model"))

def test_model_loading():
    """Test if LSTM models can be loaded successfully."""
    print("Testing model loading...")
    
    try:
        from prediction_utils import GLucosePredictionModel
        
        predictor = GLucosePredictionModel()
        success = predictor.load_models()
        
        if success:
            print("✓ Models loaded successfully!")
            return True
        else:
            print("✗ Failed to load models")
            return False
            
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

def test_prediction():
    """Test making predictions with sample data."""
    print("\nTesting predictions...")
    
    try:
        from prediction_utils import GLucosePredictionModel, create_sample_data
        
        # Initialize predictor
        predictor = GLucosePredictionModel()
        if not predictor.load_models():
            print("✗ Cannot test predictions - models not loaded")
            return False
        
        # Create sample data
        sample_glucose = [120, 125, 130, 135, 140, 145, 150, 155, 160, 
                         165, 170, 175, 180, 185, 190, 195, 200, 205]
        
        sample_data = create_sample_data(sample_glucose)
        
        # Test both horizons
        predictions = predictor.predict_both_horizons(sample_data)
        
        if predictions['30min'] is not None and predictions['60min'] is not None:
            print(f"✓ 30-minute prediction: {predictions['30min']:.1f} mg/dL")
            print(f"✓ 60-minute prediction: {predictions['60min']:.1f} mg/dL")
            return True
        else:
            print("✗ Predictions returned None")
            return False
            
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return False

def test_api_endpoints():
    """Test if API endpoints are available."""
    print("\nTesting API endpoints...")
    
    try:
        from app import create_app
        
        app = create_app()
        
        with app.test_client() as client:
            # Test health check
            response = client.get('/api/v1/predict/health')
            if response.status_code == 200:
                print("✓ Health check endpoint working")
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
            
            # Test model info
            response = client.get('/api/v1/predict/info')
            if response.status_code == 200:
                print("✓ Model info endpoint working")
            else:
                print(f"✗ Model info failed: {response.status_code}")
                return False
            
            # Test sample prediction
            response = client.get('/api/v1/predict/sample')
            if response.status_code == 200:
                print("✓ Sample prediction endpoint working")
            else:
                print(f"✗ Sample prediction failed: {response.status_code}")
                return False
            
            return True
            
    except Exception as e:
        print(f"✗ Error testing API endpoints: {e}")
        return False

def main():
    """Run all tests."""
    print("BuhAI-Backend LSTM Integration Test")
    print("=" * 40)
    
    tests = [
        test_model_loading,
        test_prediction,
        test_api_endpoints
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! LSTM integration successful.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
"""
API Blueprint for LSTM glucose predictions.

This module provides REST endpoints for making glucose predictions using
the trained LSTM models.
"""

from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import sys
import os
from pathlib import Path

# Add the model directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "model"))

from prediction_utils import GLucosePredictionModel, create_sample_data

# Create blueprint
prediction_bp = Blueprint('prediction', __name__)

# Global model instance (initialized once)
predictor = None

def initialize_predictor():
    """Initialize the prediction model."""
    global predictor
    if predictor is None:
        predictor = GLucosePredictionModel()
        if not predictor.load_models():
            print("Warning: Failed to load LSTM models")
            predictor = None
    return predictor is not None

def fallback_prediction(glucose_readings: List[float], horizon: str = "both") -> Dict[str, Any]:
    """
    Generate fallback predictions using simple heuristics when LSTM models are not available.
    
    Args:
        glucose_readings: List of recent glucose readings
        horizon: Prediction horizon ("30min", "60min", or "both")
    
    Returns:
        Dictionary with prediction results
    """
    if not glucose_readings:
        return {
            "error": "No glucose readings provided",
            "message": "Need glucose readings for fallback prediction"
        }
    
    current_glucose = glucose_readings[-1]
    
    # Simple trend analysis
    if len(glucose_readings) >= 3:
        recent_trend = (glucose_readings[-1] - glucose_readings[-3]) / 2
    else:
        recent_trend = 0
    
    # Add some realistic variation
    variation = 5 + abs(recent_trend) * 2
    noise = np.random.normal(0, 3)
    
    # Predict based on trend with bounds
    pred_30 = max(70, min(250, current_glucose + recent_trend * 6 + noise))
    pred_60 = max(70, min(250, current_glucose + recent_trend * 12 + noise))
    
    result = {
        "success": True,
        "message": "Fallback prediction generated (LSTM models unavailable)",
        "fallback": True
    }
    
    if horizon == "30min":
        result["predictions"] = {"30min": round(pred_30)}
    elif horizon == "60min":
        result["predictions"] = {"60min": round(pred_60)}
    else:  # both
        result["predictions"] = {
            "30min": round(pred_30),
            "60min": round(pred_60)
        }
    
    return result

@prediction_bp.route('/predict', methods=['POST'])
def predict_glucose():
    """
    Predict glucose levels for 30-minute and 60-minute horizons.
    
    Expected JSON payload:
    {
        "glucose_readings": [float, ...],  # List of recent glucose readings
        "meal_flags": [int, ...],          # Optional: meal flags (0 or 1)
        "activity_levels": [int, ...],     # Optional: activity levels (0, 1, 2)
        "sleep_quality": int,              # Optional: sleep quality (0 or 1)
        "horizon": "30min" or "60min"      # Optional: specific horizon
    }
    
    Returns:
        JSON response with predictions
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'glucose_readings' not in data:
            return jsonify({
                "error": "Missing required data",
                "message": "glucose_readings is required"
            }), 400
        
        glucose_readings = data['glucose_readings']
        meal_flags = data.get('meal_flags', None)
        activity_levels = data.get('activity_levels', None)
        sleep_quality = data.get('sleep_quality', 0)
        horizon = data.get('horizon', 'both')
        
        # Validate glucose readings
        if not isinstance(glucose_readings, list) or len(glucose_readings) < 1:
            return jsonify({
                "error": "Invalid glucose readings",
                "message": "Need at least 1 glucose reading for prediction"
            }), 400
        
        # Try to initialize predictor
        if not initialize_predictor():
            print("LSTM models not available, using fallback prediction")
            return jsonify(fallback_prediction(glucose_readings, horizon))
        
        # Validate glucose readings for LSTM
        if len(glucose_readings) < 18:
            print("Insufficient data for LSTM, using fallback prediction")
            return jsonify(fallback_prediction(glucose_readings, horizon))
        
        # Create sample data
        sample_data = create_sample_data(
            glucose_readings,
            meal_flags,
            activity_levels,
            sleep_quality
        )
        
        # Make predictions using LSTM
        if horizon == "30min":
            predictions = {"30min": predictor.predict_30_min(sample_data)}
        elif horizon == "60min":
            predictions = {"60min": predictor.predict_60_min(sample_data)}
        else:  # both
            predictions = predictor.predict_both_horizons(sample_data)
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "message": "LSTM predictions generated successfully"
        })
        
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        # Fallback to simple prediction on any error
        glucose_readings = data.get('glucose_readings', [120]) if 'data' in locals() else [120]
        horizon = data.get('horizon', 'both') if 'data' in locals() else 'both'
        return jsonify(fallback_prediction(glucose_readings, horizon))

@prediction_bp.route('/predict/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for the prediction service.
    
    Returns:
        JSON response with service status
    """
    try:
        model_loaded = initialize_predictor()
        
        return jsonify({
            "service": "LSTM Glucose Prediction",
            "status": "healthy" if model_loaded else "degraded",
            "models_loaded": model_loaded,
            "message": "Service is running" if model_loaded else "Models not loaded, using fallback",
            "fallback_available": True
        })
        
    except Exception as e:
        return jsonify({
            "service": "LSTM Glucose Prediction",
            "status": "degraded",
            "error": str(e),
            "fallback_available": True
        }), 200  # Still return 200 since fallback is available

@prediction_bp.route('/predict/info', methods=['GET'])
def model_info():
    """
    Get information about the prediction models.
    
    Returns:
        JSON response with model information
    """
    try:
        model_loaded = initialize_predictor()
        
        return jsonify({
            "model_type": "LSTM" if model_loaded else "Fallback",
            "features": [
                "glucose_level",
                "meal_flag_hiGI",
                "activity_intensity",
                "sleep_quality",
                "hour_sin",
                "hour_cos"
            ] if model_loaded else ["glucose_level"],
            "prediction_horizons": ["30min", "60min"],
            "input_requirements": {
                "min_readings": 18 if model_loaded else 1,
                "reading_interval": "5 minutes",
                "history_duration": "90 minutes" if model_loaded else "Variable"
            },
            "output_units": "mg/dL",
            "lstm_available": model_loaded,
            "fallback_available": True
        })
        
    except Exception as e:
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500

@prediction_bp.route('/predict/sample', methods=['GET'])
def sample_prediction():
    """
    Make a sample prediction using dummy data for testing.
    
    Returns:
        JSON response with sample prediction
    """
    try:
        # Create sample glucose readings (18 readings for 90 minutes)
        sample_glucose = [120, 125, 130, 135, 140, 145, 150, 155, 160, 
                         165, 170, 175, 180, 185, 190, 195, 200, 205]
        
        # Try to initialize predictor
        if not initialize_predictor():
            print("LSTM models not available, using fallback for sample prediction")
            result = fallback_prediction(sample_glucose, "both")
            result["sample_data"] = {
                "glucose_readings": sample_glucose,
                "description": "Sample glucose readings over 90 minutes"
            }
            return jsonify(result)
        
        # Create sample data
        sample_data = create_sample_data(sample_glucose)
        
        # Make predictions
        predictions = predictor.predict_both_horizons(sample_data)
        
        return jsonify({
            "success": True,
            "sample_data": {
                "glucose_readings": sample_glucose,
                "description": "Sample glucose readings over 90 minutes"
            },
            "predictions": predictions,
            "message": "Sample prediction generated successfully"
        })
        
    except Exception as e:
        print(f"Error in sample prediction: {e}")
        # Fallback to simple prediction
        sample_glucose = [120, 125, 130, 135, 140, 145, 150, 155, 160, 
                         165, 170, 175, 180, 185, 190, 195, 200, 205]
        result = fallback_prediction(sample_glucose, "both")
        result["sample_data"] = {
            "glucose_readings": sample_glucose,
            "description": "Sample glucose readings over 90 minutes (fallback)"
        }
        return jsonify(result) 
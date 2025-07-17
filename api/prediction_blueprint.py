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
        # Initialize predictor if needed
        if not initialize_predictor():
            return jsonify({
                "error": "LSTM models not available",
                "message": "Models failed to load"
            }), 500
        
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
        if not isinstance(glucose_readings, list) or len(glucose_readings) < 18:
            return jsonify({
                "error": "Invalid glucose readings",
                "message": "Need at least 18 glucose readings for prediction"
            }), 400
        
        # Create sample data
        sample_data = create_sample_data(
            glucose_readings,
            meal_flags,
            activity_levels,
            sleep_quality
        )
        
        # Make predictions
        if horizon == "30min":
            prediction = predictor.predict_glucose(sample_data, "30min")
            result = {
                "30min": prediction
            }
        elif horizon == "60min":
            prediction = predictor.predict_glucose(sample_data, "60min")
            result = {
                "60min": prediction
            }
        else:
            # Default: both horizons
            result = predictor.predict_both_horizons(sample_data)
        
        # Check for prediction errors
        if horizon != "both" and result[horizon] is None:
            return jsonify({
                "error": "Prediction failed",
                "message": "Unable to generate prediction"
            }), 500
        
        return jsonify({
            "success": True,
            "predictions": result,
            "message": "Predictions generated successfully"
        })
        
    except Exception as e:
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500

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
            "message": "Service is running" if model_loaded else "Models not loaded"
        })
        
    except Exception as e:
        return jsonify({
            "service": "LSTM Glucose Prediction",
            "status": "unhealthy",
            "error": str(e)
        }), 500

@prediction_bp.route('/predict/info', methods=['GET'])
def model_info():
    """
    Get information about the prediction models.
    
    Returns:
        JSON response with model information
    """
    try:
        return jsonify({
            "model_type": "LSTM",
            "features": [
                "glucose_level",
                "meal_flag_hiGI",
                "activity_intensity",
                "sleep_quality",
                "hour_sin",
                "hour_cos"
            ],
            "prediction_horizons": ["30min", "60min"],
            "input_requirements": {
                "min_readings": 18,
                "reading_interval": "5 minutes",
                "history_duration": "90 minutes"
            },
            "output_units": "mg/dL"
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
        # Initialize predictor if needed
        if not initialize_predictor():
            return jsonify({
                "error": "LSTM models not available",
                "message": "Models failed to load"
            }), 500
        
        # Create sample glucose readings (18 readings for 90 minutes)
        sample_glucose = [120, 125, 130, 135, 140, 145, 150, 155, 160, 
                         165, 170, 175, 180, 185, 190, 195, 200, 205]
        
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
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500 
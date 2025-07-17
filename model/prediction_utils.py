"""
Prediction utility module for LSTM glucose prediction models.

This module provides functions to load trained LSTM models and make glucose predictions
for 30-minute and 60-minute horizons.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# --- Constants ---
MODELS_DIR = Path(__file__).parent / "models"
SCALER_PATH = MODELS_DIR / "scaler_mv.pkl"
MODEL_30_MIN_PATH = MODELS_DIR / "lstm_mv_30.h5"
MODEL_60_MIN_PATH = MODELS_DIR / "lstm_mv_60.h5"

# Model configuration
N_PAST_READINGS = 18  # 90 minutes of history (18 readings * 5 min/reading)
HORIZON_30_MIN = 6  # 30 minutes ahead (6 steps * 5 min/step)
HORIZON_60_MIN = 12  # 60 minutes ahead (12 steps * 5 min/step)

# Feature order (must match training data)
FEATURES = [
    "glucose_level",
    "meal_flag_hiGI",
    "activity_intensity",
    "sleep_quality",
    "hour_sin",
    "hour_cos",
]


class GLucosePredictionModel:
    """
    A class for loading and using trained LSTM models for glucose prediction.
    """
    
    def __init__(self):
        """Initialize the prediction model."""
        self.scalers: Optional[Dict[str, MinMaxScaler]] = None
        self.model_30_min = None
        self.model_60_min = None
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """
        Load the trained LSTM models and scalers.
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            # Load scalers
            self.scalers = joblib.load(SCALER_PATH)
            
            # Load models
            self.model_30_min = load_model(MODEL_30_MIN_PATH)
            self.model_60_min = load_model(MODEL_60_MIN_PATH)
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def prepare_features(self, recent_data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare features for prediction from recent glucose and context data.
        
        Args:
            recent_data: DataFrame with columns matching FEATURES
            
        Returns:
            np.ndarray: Scaled features ready for prediction, or None if error
        """
        if not self.is_loaded:
            print("Models not loaded. Call load_models() first.")
            return None
            
        try:
            # Ensure we have the required features
            if not all(col in recent_data.columns for col in FEATURES):
                missing = [col for col in FEATURES if col not in recent_data.columns]
                print(f"Missing required features: {missing}")
                return None
            
            # Take the last N_PAST_READINGS rows
            if len(recent_data) < N_PAST_READINGS:
                print(f"Need at least {N_PAST_READINGS} data points, got {len(recent_data)}")
                return None
                
            recent_data = recent_data.tail(N_PAST_READINGS)
            
            # Scale features
            scaled_data = np.zeros((N_PAST_READINGS, len(FEATURES)))
            for i, feature in enumerate(FEATURES):
                scaler = self.scalers[feature]
                scaled_data[:, i] = scaler.transform(recent_data[[feature]]).flatten()
            
            # Reshape for LSTM input (1, timesteps, features)
            return scaled_data.reshape(1, N_PAST_READINGS, len(FEATURES))
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def predict_glucose(self, recent_data: pd.DataFrame, horizon: str = "30min") -> Optional[float]:
        """
        Predict glucose level for the specified horizon.
        
        Args:
            recent_data: DataFrame with recent glucose and context data
            horizon: Either "30min" or "60min"
            
        Returns:
            float: Predicted glucose level in mg/dL, or None if error
        """
        if not self.is_loaded:
            print("Models not loaded. Call load_models() first.")
            return None
            
        try:
            # Prepare features
            features = self.prepare_features(recent_data)
            if features is None:
                return None
            
            # Select appropriate model
            if horizon == "30min":
                model = self.model_30_min
            elif horizon == "60min":
                model = self.model_60_min
            else:
                print(f"Invalid horizon: {horizon}. Use '30min' or '60min'")
                return None
            
            # Make prediction
            prediction_scaled = model.predict(features, verbose=0)
            
            # Inverse transform to get actual glucose value
            glucose_scaler = self.scalers["glucose_level"]
            glucose_predicted = glucose_scaler.inverse_transform(prediction_scaled)[0][0]
            
            return float(glucose_predicted)
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_both_horizons(self, recent_data: pd.DataFrame) -> Dict[str, Optional[float]]:
        """
        Predict glucose levels for both 30-minute and 60-minute horizons.
        
        Args:
            recent_data: DataFrame with recent glucose and context data
            
        Returns:
            Dict with predictions for both horizons
        """
        return {
            "30min": self.predict_glucose(recent_data, "30min"),
            "60min": self.predict_glucose(recent_data, "60min")
        }


def create_sample_data(glucose_values: List[float], 
                      meal_flags: List[int] = None,
                      activity_levels: List[int] = None,
                      sleep_quality: int = 0) -> pd.DataFrame:
    """
    Create sample data for testing predictions.
    
    Args:
        glucose_values: List of recent glucose readings
        meal_flags: List of meal flags (0 or 1)
        activity_levels: List of activity intensity levels (0, 1, 2)
        sleep_quality: Sleep quality value (0 or 1)
        
    Returns:
        pd.DataFrame: Sample data for prediction
    """
    n_points = len(glucose_values)
    
    # Create timestamp range (assuming 5-minute intervals)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq='5T')
    
    # Default values if not provided
    if meal_flags is None:
        meal_flags = [0] * n_points
    if activity_levels is None:
        activity_levels = [0] * n_points
        
    # Create cyclical time features
    hours = [ts.hour for ts in timestamps]
    hour_sin = [np.sin(2 * np.pi * h / 24) for h in hours]
    hour_cos = [np.cos(2 * np.pi * h / 24) for h in hours]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'glucose_level': glucose_values,
        'meal_flag_hiGI': meal_flags,
        'activity_intensity': activity_levels,
        'sleep_quality': [sleep_quality] * n_points,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos
    })


# Example usage
if __name__ == "__main__":
    # Initialize prediction model
    predictor = GLucosePredictionModel()
    
    # Load models
    if predictor.load_models():
        print("Models loaded successfully!")
        
        # Create sample data (18 glucose readings for 90 minutes of history)
        sample_glucose = [120, 125, 130, 135, 140, 145, 150, 155, 160, 
                         165, 170, 175, 180, 185, 190, 195, 200, 205]
        
        sample_data = create_sample_data(sample_glucose)
        
        # Make predictions
        predictions = predictor.predict_both_horizons(sample_data)
        
        print(f"30-minute prediction: {predictions['30min']:.1f} mg/dL")
        print(f"60-minute prediction: {predictions['60min']:.1f} mg/dL")
        
    else:
        print("Failed to load models!") 
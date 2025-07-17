"""Trains multivariate time series models to predict CGM glucose levels."""
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.models import Sequential, load_model

warnings.filterwarnings("ignore")

# --- Constants ---
DATA_DIR = Path("../data/dataset-user")
MODELS_DIR = Path("models")
CGM_FILE = DATA_DIR / "cgm_stream.csv"
MEAL_FILE = DATA_DIR / "meal_events.csv"
ACTIVITY_FILE = DATA_DIR / "activity_logs.csv"
SLEEP_FILE = DATA_DIR / "sleep_logs.csv"

SCALER_PATH = MODELS_DIR / "scaler_mv.pkl"
MODEL_30_MIN_PATH = MODELS_DIR / "lstm_mv_30.h5"
MODEL_60_MIN_PATH = MODELS_DIR / "lstm_mv_60.h5"

# Sequence and Horizon Configuration
N_PAST_READINGS = 18  # 90 minutes of history (18 readings * 5 min/reading)
HORIZON_30_MIN = 6  # 30 minutes ahead (6 steps * 5 min/step)
HORIZON_60_MIN = 12  # 60 minutes ahead (12 steps * 5 min/step)

# Data Split Ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1  # Train: 70%, Val: 10%, Test: 20%

# Model Training Hyperparameters
LSTM_UNITS = 64
GRU_UNITS = 64
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 4
FEATURES = [
    "glucose_level",
    "meal_flag_hiGI",
    "activity_intensity",
    "sleep_quality",
    "hour_sin",
    "hour_cos",
]
TARGET = "glucose_level"


def create_features(cgm_file: Path, meal_file: Path, activity_file: Path, sleep_file: Path) -> pd.DataFrame:
    """Loads all data sources and engineers features for the multivariate model."""
    # 1. Load and resample CGM data to a consistent 5-minute frequency
    df_cgm = pd.read_csv(cgm_file, usecols=["timestamp", "glucose_level"], parse_dates=["timestamp"])
    df_cgm.sort_values("timestamp", inplace=True)
    df_cgm = df_cgm.set_index("timestamp").resample("5T").mean()
    df_cgm["glucose_level"] = df_cgm["glucose_level"].interpolate(method="linear")
    df_cgm.reset_index(inplace=True)

    # 2. Engineer meal feature: flag for high-GI meal in the last 90 mins
    df_meal = pd.read_csv(meal_file, usecols=["timestamp", "next_hyper_risk"], parse_dates=["timestamp"])
    df_meal_high_gi = df_meal[df_meal["next_hyper_risk"] == 1].copy()
    df_meal_high_gi["meal_flag_hiGI"] = 1
    df_meal_high_gi = df_meal_high_gi.set_index("timestamp")[["meal_flag_hiGI"]]

    meal_flags_reindexed = df_meal_high_gi.reindex(df_cgm.set_index("timestamp").index, fill_value=0)
    df_cgm["meal_flag_hiGI"] = meal_flags_reindexed.rolling(window=f"{N_PAST_READINGS*5}min", min_periods=1).max().values
    df_cgm["meal_flag_hiGI"].fillna(0, inplace=True)

    # 3. Engineer activity feature: intensity of overlapping activity
    df_activity = pd.read_csv(activity_file, parse_dates=["timestamp_start"])
    intensity_map = {"low": 0, "medium": 1, "high": 2}
    df_activity["activity_intensity"] = df_activity["intensity"].map(intensity_map)
    df_cgm["activity_intensity"] = 0.0

    # Create an interval-based mapping for activities
    for _, row in df_activity.iterrows():
        start, duration, intensity = row["timestamp_start"], row["duration_min"], row["activity_intensity"]
        end = start + pd.to_timedelta(duration, unit="m")
        df_cgm.loc[df_cgm["timestamp"].between(start, end), "activity_intensity"] = intensity

    # 4. Engineer sleep feature: forward-filled quality from last night's sleep
    df_sleep = pd.read_csv(sleep_file, parse_dates=["sleep_start", "sleep_end"])
    quality_map = {"good": 0, "poor": 1}
    df_sleep["sleep_quality_val"] = df_sleep["sleep_quality"].map(quality_map)
    df_sleep["date"] = df_sleep["sleep_end"].dt.date
    daily_quality = df_sleep.drop_duplicates(subset="date", keep="last").set_index("date")["sleep_quality_val"]
    
    df_cgm["date"] = df_cgm["timestamp"].dt.date
    df_cgm = pd.merge(df_cgm, daily_quality.rename("sleep_quality"), on="date", how="left")
    df_cgm["sleep_quality"].ffill(inplace=True)
    df_cgm["sleep_quality"].bfill(inplace=True) # Fill any initial NaNs

    # 5. Engineer cyclical time features
    df_cgm["hour"] = df_cgm["timestamp"].dt.hour
    df_cgm["hour_sin"] = np.sin(2 * np.pi * df_cgm["hour"] / 24)
    df_cgm["hour_cos"] = np.cos(2 * np.pi * df_cgm["hour"] / 24)

    # Finalize
    df_cgm.set_index("timestamp", inplace=True)
    final_df = df_cgm[FEATURES].copy()
    final_df.dropna(inplace=True) # Drop rows with NaNs from initial interpolation
    return final_df


def create_sequences(
    data: np.ndarray, n_past: int, n_future: int, target_col_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates supervised learning sequences from multivariate time series data."""
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_past
        out_end_ix = end_ix + n_future - 1
        if out_end_ix >= len(data):
            break
        X.append(data[i:end_ix, :])
        y.append(data[out_end_ix, target_col_idx])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """Builds and compiles a 2-layer LSTM model."""
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        LSTM(LSTM_UNITS),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_gru_model(input_shape: Tuple[int, int]) -> Sequential:
    """Builds and compiles a 1-layer GRU model."""
    model = Sequential([
        GRU(GRU_UNITS, input_shape=input_shape),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_select_model(
    train_data: np.ndarray,
    val_data: np.ndarray,
    target_idx: int,
    n_past: int,
    n_future: int,
) -> Tuple[Sequential, str]:
    """Trains both LSTM and GRU models, and selects the one with the lower validation loss."""
    X_train, y_train = create_sequences(train_data, n_past, n_future, target_idx)
    X_val, y_val = create_sequences(val_data, n_past, n_future, target_idx)

    input_shape = (X_train.shape[1], X_train.shape[2])
    early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    print("--- Training LSTM model... ---")
    lstm_model = build_lstm_model(input_shape)
    lstm_history = lstm_model.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val), callbacks=[early_stopper], verbose=1
    )
    lstm_val_loss = min(lstm_history.history['val_loss'])
    print(f"LSTM best validation loss: {lstm_val_loss:.4f}")

    print("--- Training GRU model... ---")
    gru_model = build_gru_model(input_shape)
    gru_history = gru_model.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val), callbacks=[early_stopper], verbose=1
    )
    gru_val_loss = min(gru_history.history['val_loss'])
    print(f"GRU best validation loss: {gru_val_loss:.4f}")

    if lstm_val_loss <= gru_val_loss:
        print("--- Selecting LSTM model ---")
        return lstm_model, "LSTM"
    else:
        print("--- Selecting GRU model ---")
        return gru_model, "GRU"


def evaluate_model(
    model: Sequential,
    data: np.ndarray,
    scalers: Dict[str, MinMaxScaler],
    target_name: str,
    target_idx: int,
    n_past: int,
    n_future: int,
    dataset_name: str,
) -> None:
    """Evaluates the model and prints MAE and RMSE."""
    X, y_true_scaled = create_sequences(data, n_past, n_future, target_idx)
    y_pred_scaled = model.predict(X)

    target_scaler = scalers[target_name]
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{dataset_name} MAE:  {mae:.2f} mg/dL")
    print(f"{dataset_name} RMSE: {rmse:.2f} mg/dL")


def main() -> None:
    """Main function to run the complete training pipeline."""
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Feature Engineering
    df = create_features(CGM_FILE, MEAL_FILE, ACTIVITY_FILE, SLEEP_FILE)
    print("Feature engineering complete. Data shape:", df.shape)
    
    # 2. Data Splitting
    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    df_train, df_val, df_test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    print(f"Data split: Train ({len(df_train)}), Validation ({len(df_val)}), Test ({len(df_test)})")

    # 3. Scaling
    scalers: Dict[str, MinMaxScaler] = {}
    train_scaled, val_scaled, test_scaled = (np.zeros_like(d.values) for d in (df_train, df_val, df_test))

    for i, col in enumerate(df.columns):
        scaler = MinMaxScaler()
        train_scaled[:, i] = scaler.fit_transform(df_train[[col]]).flatten()
        val_scaled[:, i] = scaler.transform(df_val[[col]]).flatten()
        test_scaled[:, i] = scaler.transform(df_test[[col]]).flatten()
        scalers[col] = scaler

    joblib.dump(scalers, SCALER_PATH)
    print(f"Scalers saved to {SCALER_PATH}")
    target_idx = df.columns.get_loc(TARGET)

    # 4. Train and evaluate for each horizon
    for horizon_steps, model_path in [(HORIZON_30_MIN, MODEL_30_MIN_PATH), (HORIZON_60_MIN, MODEL_60_MIN_PATH)]:
        horizon_min = horizon_steps * 5
        print(f"\n--- Training for {horizon_min}-minute horizon ---")

        best_model, model_type = train_and_select_model(
            train_scaled, val_scaled, target_idx, N_PAST_READINGS, horizon_steps
        )
        print(f"\n--- Evaluating {model_type} for {horizon_min}-minute horizon ---")
        evaluate_model(best_model, train_scaled, scalers, TARGET, target_idx, N_PAST_READINGS, horizon_steps, "Train")
        evaluate_model(best_model, val_scaled, scalers, TARGET, target_idx, N_PAST_READINGS, horizon_steps, "Validation")
        evaluate_model(best_model, test_scaled, scalers, TARGET, target_idx, N_PAST_READINGS, horizon_steps, "Test")
        
        best_model.save(model_path)
        print(f"Best model ({model_type}) saved to {model_path}")


if __name__ == "__main__":
    main() 
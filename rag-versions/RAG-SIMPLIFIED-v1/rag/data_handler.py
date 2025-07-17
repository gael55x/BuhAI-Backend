import logging
from datetime import datetime, timedelta, date
import random
import re

import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func

from db.models import Base, CGMStream, MealEvent, ActivityLog, SleepLog, CGMAggregate

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DB_PATH = "data/buhai.db"
HIGH_GI_KEYWORDS = ["rice", "bread", "soda", "cake", "sweet", "chocolate", "kan-on", "tinapay"]

def is_high_gi(food_description: str) -> bool:
    """Simplified check for high-glycemic index foods based on keywords."""
    if not isinstance(food_description, str):
        return False
    return any(keyword in food_description.lower() for keyword in HIGH_GI_KEYWORDS)

class DataHandler:
    def __init__(self, db_path=DB_PATH):
        """Initializes the data handler by connecting to the database."""
        try:
            engine = create_engine(f'sqlite:///{db_path}')
            Base.metadata.bind = engine
            DBSession = sessionmaker(bind=engine)
            self.session = DBSession()
            logger.info("Database session created successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to database at {db_path}: {e}", exc_info=True)
            self.session = None

    def get_last_log_timestamp(self) -> datetime:
        """
        Finds the latest timestamp from all user-generated log tables.

        Returns:
            datetime: The most recent timestamp found, or None if no logs exist.
        """
        if not self.session:
            logger.error("Cannot fetch data: no database session.")
            return None

        # Query for the max timestamp in each relevant table
        last_meal_time = self.session.query(func.max(MealEvent.timestamp)).scalar()
        last_activity_time = self.session.query(func.max(ActivityLog.timestamp_start)).scalar()
        last_sleep_time = self.session.query(func.max(SleepLog.sleep_start)).scalar()

        # Filter out None values and find the overall maximum
        timestamps = [t for t in [last_meal_time, last_activity_time, last_sleep_time] if t is not None]

        if not timestamps:
            logger.warning("No user logs found in the database.")
            return None

        return max(timestamps)

    def get_last_meal_timestamp(self) -> datetime:
        """
        Finds the latest timestamp from the meal_events table.

        Returns:
            datetime: The most recent meal timestamp, or None if no meal logs exist.
        """
        if not self.session:
            logger.error("Cannot fetch data: no database session.")
            return None

        last_meal_time = self.session.query(func.max(MealEvent.timestamp)).scalar()
        return last_meal_time

    def get_daily_summary(self, target_date: date) -> dict:
        """
        Fetches the aggregated CGM summary for a specific date and the day before.

        Args:
            target_date (date): The primary date for the summary.

        Returns:
            dict: A dictionary containing summary stats for the target date and the previous day.
        """
        if not self.session:
            logger.error("Cannot fetch data: no database session.")
            return None
        
        previous_date = target_date - timedelta(days=1)

        summary_today = self.session.query(CGMAggregate).filter(
            CGMAggregate.date == target_date
        ).first()

        summary_yesterday = self.session.query(CGMAggregate).filter(
            CGMAggregate.date == previous_date
        ).first()

        return {
            "today": summary_today,
            "yesterday": summary_yesterday
        }

    def get_prediction_features(self, end_time: datetime, window_minutes: int = 90) -> pd.DataFrame:
        """
        Fetches and prepares the feature DataFrame needed for an ML prediction.

        Args:
            end_time (datetime): The timestamp for which to make a prediction (e.g., now).
            window_minutes (int): The duration of historical data to fetch.

        Returns:
            pd.DataFrame: A DataFrame with the required features, resampled to 5-minute intervals.
                          Returns an empty DataFrame on error.
        """
        if not self.session:
            logger.error("Cannot fetch data: no database session.")
            return pd.DataFrame()

        start_time = end_time - timedelta(minutes=window_minutes)
        
        # 1. Fetch and process CGM data
        cgm_query = self.session.query(CGMStream).filter(
            CGMStream.timestamp.between(start_time, end_time)
        ).statement
        df_cgm = pd.read_sql(cgm_query, self.session.bind)
        if df_cgm.empty:
            logger.warning("No CGM data found in the specified window.")
            return pd.DataFrame()
            
        df_cgm['timestamp'] = pd.to_datetime(df_cgm['timestamp'])
        df_cgm = df_cgm[['timestamp', 'glucose_level']].set_index('timestamp').resample('5T').mean()
        df_cgm['glucose_level'] = df_cgm['glucose_level'].interpolate(method='linear')
        
        # 2. Fetch and process Meal data for high-GI flag
        meal_query = self.session.query(MealEvent).filter(
            MealEvent.timestamp.between(start_time, end_time)
        ).statement
        df_meal = pd.read_sql(meal_query, self.session.bind)
        df_meal['timestamp'] = pd.to_datetime(df_meal['timestamp'])
        df_meal['meal_flag_hiGI'] = df_meal['food_items'].apply(is_high_gi).astype(int)
        
        df_meal_flags = df_meal.set_index('timestamp')[['meal_flag_hiGI']]
        meal_flags_reindexed = df_meal_flags.reindex(df_cgm.index, fill_value=0)
        df_cgm['meal_flag_hiGI'] = meal_flags_reindexed.rolling(window=f"{window_minutes}min", min_periods=1).max()
        df_cgm['meal_flag_hiGI'].fillna(0, inplace=True)

        # 3. Fetch and process Activity data
        activity_query = self.session.query(ActivityLog).filter(
             ActivityLog.timestamp_start.between(start_time - timedelta(minutes=60), end_time) # Widen search
        ).statement
        df_activity = pd.read_sql(activity_query, self.session.bind)
        intensity_map = {"low": 0, "medium": 1, "high": 2}
        df_activity["activity_intensity"] = df_activity["intensity"].map(intensity_map)
        df_activity['timestamp_start'] = pd.to_datetime(df_activity['timestamp_start'])
        df_cgm["activity_intensity"] = 0.0

        for _, row in df_activity.iterrows():
            act_start = row["timestamp_start"]
            act_end = act_start + pd.to_timedelta(row["duration_min"], unit="m")
            df_cgm.loc[df_cgm.index.between(act_start, act_end), "activity_intensity"] = row["activity_intensity"]

        # 4. Fetch and process Sleep data
        sleep_query = self.session.query(SleepLog).filter(
            SleepLog.sleep_end.between(end_time - timedelta(days=2), end_time)
        ).statement
        df_sleep = pd.read_sql(sleep_query, self.session.bind)
        if not df_sleep.empty:
            quality_map = {"good": 0, "poor": 1}
            last_sleep = df_sleep.sort_values('sleep_end').iloc[-1]
            df_cgm['sleep_quality'] = quality_map.get(last_sleep['sleep_quality'], 0.5) # Default to neutral
        else:
            df_cgm['sleep_quality'] = 0.5 # Default if no sleep data
            
        # 5. Engineer time features
        df_cgm["hour"] = df_cgm.index.hour
        df_cgm["hour_sin"] = np.sin(2 * np.pi * df_cgm["hour"] / 24)
        df_cgm["hour_cos"] = np.cos(2 * np.pi * df_cgm["hour"] / 24)

        # Finalize and ensure correct column order
        features_columns = [
            "glucose_level", "meal_flag_hiGI", "activity_intensity", 
            "sleep_quality", "hour_sin", "hour_cos"
        ]
        final_df = df_cgm[features_columns].copy()
        final_df.ffill(inplace=True) # Fill any gaps from resampling
        final_df.bfill(inplace=True)
        
        if final_df.isnull().values.any():
            logger.warning("Feature DataFrame contains NaNs after processing. Filling with 0.")
            final_df.fillna(0, inplace=True)

        return final_df

    def get_historical_meal_estimate(self, food_description: str) -> float:
        """
        Provides a simple glucose estimate based on historical averages for similar meals,
        factoring in the quantity mentioned in the description.

        Args:
            food_description (str): The description of the meal (e.g., "2 cups of rice").

        Returns:
            float: An estimated glucose value, or a default value if no history is found.
        """
        if not self.session:
            logger.error("Cannot fetch data: no database session.")
            return 140.0 + random.uniform(-2, 2)

        # --- Quantity Parsing ---
        quantity_match = re.search(r'\b(\d+(\.\d+)?)\b', food_description)
        quantity = float(quantity_match.group(1)) if quantity_match else 1.0
        # Basic sanity check for quantity
        if quantity > 10: # Cap quantity to avoid extreme values from faulty parsing
            quantity = 10
        
        logger.info(f"Parsed quantity: {quantity} from description: '{food_description}'")


        # Find keywords in the new meal description
        keywords = [word for word in HIGH_GI_KEYWORDS if word in food_description.lower()]
        
        # Base estimate for a single unit of a high-GI meal
        base_high_gi_estimate = 165.0

        if not keywords:
            # If no high-GI keywords, assume it's a standard meal. Quantity has less impact.
            logger.info("No high-GI keywords found. Returning a default low-GI estimate.")
            # Let's make quantity have a smaller effect on low-GI foods
            return 120.0 + (quantity - 1) * 5 + random.uniform(-2, 2)

        # Build a query to find past meals with similar keywords
        query = self.session.query(MealEvent)
        for keyword in keywords:
            query = query.filter(MealEvent.food_items.like(f'%{keyword}%'))
        
        past_meals = query.limit(10).all()

        if not past_meals:
            logger.warning(f"No similar past meals found for '{food_description}'. Using scaled default estimate.")
            # Scale the default high-GI estimate by the parsed quantity
            return base_high_gi_estimate * quantity * 0.5 + random.uniform(-2, 2) #use 0.5 as factor

        # For each similar meal, find the CGM reading ~30-60 minutes after
        glucose_peaks = []
        for meal in past_meals:
            meal_time = meal.timestamp
            post_meal_start = meal_time + timedelta(minutes=30)
            post_meal_end = meal_time + timedelta(minutes=60)
            
            # Find the max glucose reading in the window after the meal
            peak_query = self.session.query(func.max(CGMStream.glucose_level)).filter(
                CGMStream.timestamp.between(post_meal_start, post_meal_end)
            )
            peak = peak_query.scalar()
            
            if peak:
                glucose_peaks.append(peak)

        if not glucose_peaks:
            logger.warning("Found similar meals, but no subsequent CGM data to create an estimate.")
            # Fallback to scaled default estimate if no CGM data is found
            return base_high_gi_estimate * quantity * 0.5 + random.uniform(-2, 2)

        # Return the average of the historical peaks, scaled by quantity
        historical_average_peak = np.mean(glucose_peaks)
        # We assume the historical average is for a "standard" portion (quantity=1)
        # The estimate is scaled based on the new quantity.
        estimated_peak = historical_average_peak + (quantity - 1) * 15 # Add 15 points per extra quantity
        logger.info(f"Historical average: {historical_average_peak:.2f}. Scaled estimate for quantity {quantity}: {estimated_peak:.2f}")

        return estimated_peak + random.uniform(-2, 2) 
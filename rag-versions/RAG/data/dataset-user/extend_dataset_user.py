
import pandas as pd
import numpy as np
import datetime
from scipy.stats import truncnorm
import os
from pathlib import Path

# --- CONFIGURATION ---
TARGET_DIR = Path('/Users/gailleamolong/Documents/Documents/RAG/data/dataset-user')
TIMEZONE = "Asia/Manila"
START_DATE = pd.Timestamp('2025-07-10 00:00', tz=TIMEZONE)
END_DATE = pd.Timestamp('2025-07-15 14:00', tz=TIMEZONE)

# --- FILIPINO FOOD DATABASE (from data_gen.py) ---
FOOD_DATABASE = {
    # High-GI
    'white_rice': {'gi': 'high'}, 'pancit_canton': {'gi': 'high'},
    'turon': {'gi': 'high'}, 'halo-halo': {'gi': 'high'},
    'milk_tea': {'gi': 'high'},
    # Moderate-GI
    'brown_rice': {'gi': 'moderate'}, 'fish_tinola': {'gi': 'moderate'},
    'chicken_adobo': {'gi': 'moderate'}, 'sinigang_na_baboy': {'gi': 'moderate'},
    'lechon_kawali': {'gi': 'moderate'},
    # Low-GI
    'itlog': {'gi': 'low'}, 'longganisa': {'gi': 'low'},
    'tapa': {'gi': 'low'}, 'grilled_tuna': {'gi': 'low'},
    'pork_giniling': {'gi': 'low'},
}

def generate_sleep_logs(days):
    """Generates sleep data using a truncated normal distribution."""
    logs = []
    # truncnorm: a, b are std devs from mean. (5-7.5)/1 = -2.5, (9-7.5)/1 = 1.5
    sleep_dist = truncnorm(a=-2.5, b=1.5, loc=7.5, scale=1)

    for day in days:
        sleep_start_base = day + pd.Timedelta(hours=23)
        sleep_start = sleep_start_base + pd.Timedelta(minutes=np.random.randint(-60, 60))
        
        sleep_duration_hours = sleep_dist.rvs(1)[0]
        sleep_end = sleep_start + pd.Timedelta(hours=sleep_duration_hours)

        # Per prompt, last sleep night (14->15 July) must end by 08:00 on July 15.
        if day.date() == (END_DATE.normalize() - pd.Timedelta(days=1)).date():
            limit = END_DATE.normalize().replace(hour=8, minute=0, second=0, microsecond=0)
            if sleep_end > limit:
                sleep_end = limit

        if np.random.rand() < 0.25: # 25% chance of a bad night
            num_wakeups = np.random.randint(4, 6)
        else:
            num_wakeups = np.random.randint(0, 4)

        duration_h = (sleep_end - sleep_start).total_seconds() / 3600
        quality = 'poor' if duration_h < 6 or num_wakeups > 3 else 'good'
        
        logs.append({
            "date": day.date(),
            "sleep_start": sleep_start,
            "sleep_end": sleep_end,
            "duration_h": round(duration_h, 2),
            "num_wakeups": num_wakeups,
            "sleep_quality": quality,
            "was_disrupted": int(num_wakeups > 0),
        })
    return pd.DataFrame(logs)

def generate_activity_logs(days, last_day_end_time=None):
    """Generates activities, ensuring at least one low and one medium/high event per day."""
    logs = []
    activities = {
        "low_intensity": {"types": ["walk", "chores"], "duration": (20, 90), "steps": (1000, 2500)},
        "medium_high_intensity": {"types": ["commute", "gym"], "duration": (30, 75), "steps": (1500, 6000)}
    }
    
    for day in days:
        is_last_day = (last_day_end_time is not None and day.date() == last_day_end_time.date())
        
        # Adjust time bounds for the partial last day
        low_intensity_hour_range = (10, last_day_end_time.hour) if is_last_day else (10, 18)
        med_high_intensity_hour_range = (7, last_day_end_time.hour) if is_last_day else (7, 20)
        midnight = day.normalize()

        # Guarantee one low-intensity event
        if low_intensity_hour_range[1] > low_intensity_hour_range[0]:
            act_type_low = np.random.choice(activities["low_intensity"]["types"])
            timestamp_start = midnight + pd.Timedelta(hours=np.random.uniform(*low_intensity_hour_range))
            
            if is_last_day:
                max_dur = activities["low_intensity"]["duration"][1]
                latest_start = last_day_end_time - pd.Timedelta(minutes=max_dur)
                timestamp_start = min(timestamp_start, latest_start)

            logs.append({
                "activity_type": act_type_low,
                "timestamp_start": timestamp_start,
                "duration_min": np.random.randint(*activities["low_intensity"]["duration"]),
                "intensity": 'low',
                "steps": np.random.randint(*activities["low_intensity"]["steps"])
            })
        
        # Guarantee one medium/high-intensity event
        if med_high_intensity_hour_range[1] > med_high_intensity_hour_range[0]:
            act_type_med_high = np.random.choice(activities["medium_high_intensity"]["types"])
            intensity = 'high' if act_type_med_high == 'gym' else 'medium'
            timestamp_start = midnight + pd.Timedelta(hours=np.random.uniform(*med_high_intensity_hour_range))

            if is_last_day:
                max_dur = activities["medium_high_intensity"]["duration"][1]
                latest_start = last_day_end_time - pd.Timedelta(minutes=max_dur)
                timestamp_start = min(timestamp_start, latest_start)

            logs.append({
                "activity_type": act_type_med_high,
                "timestamp_start": timestamp_start,
                "duration_min": np.random.randint(*activities["medium_high_intensity"]["duration"]),
                "intensity": intensity,
                "steps": np.random.randint(*activities["medium_high_intensity"]["steps"])
            })

    if not logs:
        return pd.DataFrame(columns=["activity_type", "timestamp_start", "duration_min", "intensity", "steps"])

    return pd.DataFrame(logs).sort_values("timestamp_start").reset_index(drop=True)


def generate_meal_events(days, sleep_df, last_day_end_time=None):
    """Generates meal events without CGM data."""
    meal_events = []
    food_items_list = list(FOOD_DATABASE.keys())
    
    for day in days:
        is_last_day = (last_day_end_time is not None and day.date() == last_day_end_time.date())
        midnight = day.normalize()
        
        meal_times = {
            'breakfast': midnight + pd.Timedelta(hours=np.random.uniform(6.5, 8.5)),
            'lunch': midnight + pd.Timedelta(hours=np.random.uniform(12, 13.5)),
            'snack': midnight + pd.Timedelta(hours=np.random.uniform(15, 16.5)),
            'dinner': midnight + pd.Timedelta(hours=np.random.uniform(18, 20)),
        }
        if np.random.rand() < 0.3:
            meal_times['late_snack'] = midnight + pd.Timedelta(hours=np.random.uniform(21.5, 22.5))
        
        for meal_type, timestamp in meal_times.items():
            # For the last partial day, only include meals before the end time
            if is_last_day and timestamp > last_day_end_time:
                continue

            num_food_items = np.random.randint(1, 4)
            chosen_foods = np.random.choice(food_items_list, num_food_items, replace=False)
            
            meal_events.append({
                "timestamp": timestamp,
                "date": timestamp.normalize().date(),
                "meal_type": 'snack' if meal_type == 'late_snack' else meal_type,
                "food_items": str(list(chosen_foods)),
                "portion_estimates": str([np.random.randint(80, 250) for _ in chosen_foods]),
            })

    if not meal_events:
        return pd.DataFrame()

    meal_df = pd.DataFrame(meal_events).sort_values("timestamp").reset_index(drop=True)
    
    # Merge sleep quality
    sleep_df['date'] = pd.to_datetime(sleep_df['date'])
    meal_df['date'] = pd.to_datetime(meal_df['date'])
    final_meal_df = meal_df.merge(sleep_df[['date', 'sleep_quality']], on='date', how='left')

    # Add blank CGM-related columns to match schema
    cgm_cols = [
        'baseline_glucose', 'glucose_at_t+30min', 'glucose_at_t+60min',
        'AUC_postprandial_2h', 'return_to_baseline_time', 'next_hypo_risk', 'next_hyper_risk'
    ]
    for col in cgm_cols:
        final_meal_df[col] = np.nan

    return final_meal_df

def main():
    """Main function to generate and append data."""
    # --- 1. READ EXISTING DATA ---
    try:
        activity_df_orig = pd.read_csv(TARGET_DIR / 'activity_logs.csv')
        meal_df_orig = pd.read_csv(TARGET_DIR / 'meal_events.csv')
        sleep_df_orig = pd.read_csv(TARGET_DIR / 'sleep_logs.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the script is run from the project root and files exist in {TARGET_DIR}.")
        return

    # --- 2. VERIFY LAST TIMESTAMPS & NORMALIZE DTYPES ---
    activity_df_orig['timestamp_start'] = pd.to_datetime(activity_df_orig['timestamp_start'], format='mixed')
    meal_df_orig['timestamp'] = pd.to_datetime(meal_df_orig['timestamp'], format='mixed')
    sleep_df_orig['sleep_start'] = pd.to_datetime(sleep_df_orig['sleep_start'], format='mixed')
    
    # Convert date columns to date objects for consistent merging/comparison
    meal_df_orig['date'] = pd.to_datetime(meal_df_orig['date'], format='mixed').dt.date
    sleep_df_orig['date'] = pd.to_datetime(sleep_df_orig['date'], format='mixed').dt.date


    last_activity_ts = activity_df_orig['timestamp_start'].max()
    if last_activity_ts.date() < (START_DATE - pd.Timedelta(days=1)).date():
        print(f"Warning: Gap in activity data. Last entry is on {last_activity_ts.date()}.")

    # --- 3. BUILD DATE RANGES ---
    # Days for which to generate full-day data
    days_to_generate = pd.date_range(start=START_DATE.normalize(), end=END_DATE.normalize(), freq='D', tz=TIMEZONE)
    # Nights for which to generate sleep data (night of 9th -> 14th)
    sleep_days = pd.date_range(start=(START_DATE - pd.Timedelta(days=1)).normalize(), end=(END_DATE - pd.Timedelta(days=1)).normalize(), freq='D', tz=TIMEZONE)

    # --- 4. GENERATE NEW DATA ---
    # Generate sleep logs first to use for meal context
    new_sleep_logs = generate_sleep_logs(sleep_days)
    new_activity_logs = generate_activity_logs(days_to_generate, last_day_end_time=END_DATE)
    new_meal_events = generate_meal_events(days_to_generate, new_sleep_logs, last_day_end_time=END_DATE)

    # --- 5. CONCATENATE AND SAVE ---
    # Re-run safety: filter out any data we might have already generated
    if not new_sleep_logs.empty:
        last_sleep_ts = sleep_df_orig['sleep_start'].max()
        new_sleep_logs = new_sleep_logs[new_sleep_logs['sleep_start'] > last_sleep_ts]
    
    if not new_activity_logs.empty:
        last_activity_ts = activity_df_orig['timestamp_start'].max()
        new_activity_logs = new_activity_logs[new_activity_logs['timestamp_start'] > last_activity_ts]

    if not new_meal_events.empty:
        last_meal_ts = meal_df_orig['timestamp'].max()
        new_meal_events = new_meal_events[new_meal_events['timestamp'] > last_meal_ts]

    appended_counts = {}

    # Activity Logs
    final_activity = pd.concat([activity_df_orig, new_activity_logs], ignore_index=True)
    final_activity.to_csv(TARGET_DIR / 'activity_logs.csv', index=False)
    appended_counts['activity'] = len(new_activity_logs)

    # Sleep Logs
    final_sleep = pd.concat([sleep_df_orig, new_sleep_logs], ignore_index=True)
    # Ensure date columns are strings without time for consistency
    final_sleep.to_csv(TARGET_DIR / 'sleep_logs.csv', index=False)
    appended_counts['sleep'] = len(new_sleep_logs)

    # Meal Events
    if not new_meal_events.empty:
        # Reorder columns to match original file exactly
        new_meal_events = new_meal_events[meal_df_orig.columns]
    final_meals = pd.concat([meal_df_orig, new_meal_events], ignore_index=True)
    final_meals.to_csv(TARGET_DIR / 'meal_events.csv', index=False)
    appended_counts['meals'] = len(new_meal_events)

    # --- 6. PRINT SUMMARY ---
    print(f"Appended rows – activity: {appended_counts['activity']}, meals: {appended_counts['meals']}, sleep: {appended_counts['sleep']}")
    print(f"Files updated in {TARGET_DIR}")

if __name__ == "__main__":
    main() 
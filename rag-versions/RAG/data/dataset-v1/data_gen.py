import pandas as pd
import numpy as np
import datetime
from faker import Faker
from dateutil.tz import gettz
from scipy.stats import norm, truncnorm, lognorm

# --- CONFIGURATION ---
NUM_DAYS = 28
TIMEZONE = "Asia/Manila"
START_DATE = pd.Timestamp.now(tz=TIMEZONE).normalize() - pd.DateOffset(days=NUM_DAYS)

USER_PROFILE = {
    "name": "John Prince Alonte",
    "age": 23,
    "sex": "male",
    "diabetes_type": "type_2",
    "years_since_diagnosis": 2,
    "is_insulin_dependent": False,
    "comorbidities": ["hypertension"],
    "medications": ["Metformin"],
    "baseline_a1c": 7.1
}

# --- FILIPINO FOOD DATABASE (per 100g) ---
# Approximate values for demonstration purposes
FOOD_DATABASE = {
    # High-GI
    'white_rice': {'gi': 'high'},
    'pancit_canton': {'gi': 'high'},
    'turon': {'gi': 'high'},
    'halo-halo': {'gi': 'high'},
    'milk_tea': {'gi': 'high'},
    # Moderate-GI
    'brown_rice': {'gi': 'moderate'},
    'fish_tinola': {'gi': 'moderate'},
    'chicken_adobo': {'gi': 'moderate'},
    'sinigang_na_baboy': {'gi': 'moderate'},
    'lechon_kawali': {'gi': 'moderate'},
    # Low-GI
    'itlog': {'gi': 'low'},
    'longganisa': {'gi': 'low'},
    'tapa': {'gi': 'low'},
    'grilled_tuna': {'gi': 'low'},
    'pork_giniling': {'gi': 'low'},
}

def apply_noise(value, percentage=0.10):
    """Applies Gaussian noise to a value."""
    return value * (1 + np.random.normal(0, percentage))

def generate_sleep_logs(days):
    """Generates 28 days of sleep data using a truncated normal distribution."""
    logs = []
    # Using a truncated normal distribution for sleep duration
    # This keeps most nights between 5 and 9 hours.
    # a, b are standard deviations from the mean.
    # (5-7.5)/1 = -2.5, (9-7.5)/1 = 1.5
    sleep_dist = truncnorm(a=-2.5, b=1.5, loc=7.5, scale=1)

    for day in days:
        sleep_start_base = day + pd.Timedelta(hours=23)
        sleep_start = sleep_start_base + pd.Timedelta(minutes=np.random.randint(-60, 60))
        
        sleep_duration_hours = sleep_dist.rvs(1)[0]
        sleep_end = sleep_start + pd.Timedelta(hours=sleep_duration_hours)
        
        # Simulating wake-ups
        if np.random.rand() < 0.25: # 25% chance of a bad night (>3 wakeups)
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

def generate_activity_logs(days):
    """Generates activities, ensuring at least one low and one medium/high intensity event per day."""
    logs = []
    activities = {
        "low_intensity": {
            "types": ["walk", "chores"],
            "duration": (20, 90),
            "steps": (1000, 2500)
        },
        "medium_high_intensity": {
            "types": ["commute", "gym"],
            "duration": (30, 75),
            "steps": (1500, 6000)
        }
    }
    
    for day in days:
        # Guarantee one low-intensity event
        act_type_low = np.random.choice(activities["low_intensity"]["types"])
        logs.append({
            "activity_type": act_type_low,
            "timestamp_start": day + pd.Timedelta(hours=np.random.uniform(10, 18)),
            "duration_min": np.random.randint(*activities["low_intensity"]["duration"]),
            "intensity": 'low',
            "steps": np.random.randint(*activities["low_intensity"]["steps"])
        })
        
        # Guarantee one medium/high-intensity event
        act_type_med_high = np.random.choice(activities["medium_high_intensity"]["types"])
        intensity = 'high' if act_type_med_high == 'gym' else 'medium'
        logs.append({
            "activity_type": act_type_med_high,
            "timestamp_start": day + pd.Timedelta(hours=np.random.uniform(7, 20)),
            "duration_min": np.random.randint(*activities["medium_high_intensity"]["duration"]),
            "intensity": intensity,
            "steps": np.random.randint(*activities["medium_high_intensity"]["steps"])
        })

    return pd.DataFrame(logs).sort_values("timestamp_start").reset_index(drop=True)

def generate_cgm_and_meal_data(days, sleep_df, activity_df):
    """
    Simulates a 28-day CGM stream and corresponding meal events with physiologically plausible dynamics.
    This function generates meals and CGM values concurrently to allow for dynamic baselines.
    """
    # 1. Setup CGM timeline and data structure
    cgm_timestamps = pd.date_range(
        start=days[0], 
        end=days[-1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=5), 
        freq='5min'
    )
    cgm_data = pd.DataFrame(index=cgm_timestamps)
    cgm_data['glucose_level'] = np.nan
    cgm_data['sensor_id'] = np.nan
    cgm_data['signal_quality_flag'] = np.nan
    
    # 2. Pre-calculate effects that don't depend on glucose history
    
    # Circadian rhythm effect (pre-calculated delta)
    hours = cgm_data.index.hour + cgm_data.index.minute / 60
    circadian_wave = 10 * np.sin(2 * np.pi * (hours - 11) / 24) # Peak at 17:00, nadir at 05:00
    circadian_effect_delta = pd.Series(circadian_wave, index=cgm_data.index).diff().fillna(0)
    
    # Activity effect (pre-calculated delta)
    activity_effect_delta = np.zeros(len(cgm_data))
    for _, activity in activity_df.iterrows():
        if activity['intensity'] in ['medium', 'high']:
            start_idx = cgm_data.index.get_indexer([activity['timestamp_start']], method='nearest')[0]
            # Effect lasts for duration + 1h taper
            duration_intervals = int((activity['duration_min'] + 60) / 5)
            slope = -0.06 if activity['intensity'] == 'high' else -0.03
            # The effect is a constant drop per 5-min interval
            activity_effect_delta[start_idx : start_idx + duration_intervals] = slope * 5
            
    # Meal impulse curve (will be built incrementally)
    meal_impulse_delta = np.zeros(len(cgm_data))
    
    # 3. Generate meal times and food choices
    meal_events = []
    food_items_list = list(FOOD_DATABASE.keys())
    gi_map = {'low': 1.0, 'moderate': 1.5, 'high': 2.0}
    
    for day in days:
        meal_times = {
            'breakfast': day + pd.Timedelta(hours=np.random.uniform(6.5, 8.5)),
            'lunch': day + pd.Timedelta(hours=np.random.uniform(12, 13.5)),
            'snack': day + pd.Timedelta(hours=np.random.uniform(15, 16.5)),
            'dinner': day + pd.Timedelta(hours=np.random.uniform(18, 20)),
        }
        if np.random.rand() < 0.3:
            meal_times['late_snack'] = day + pd.Timedelta(hours=np.random.uniform(21.5, 22.5))
            
        for meal_type, timestamp in meal_times.items():
            num_food_items = np.random.randint(1, 4)
            chosen_foods = np.random.choice(food_items_list, num_food_items, replace=False)
            
            gi_levels = [FOOD_DATABASE[food]['gi'] for food in chosen_foods]
            # Get the highest GI level from the meal
            max_gi = 'low'
            if 'high' in gi_levels: max_gi = 'high'
            elif 'moderate' in gi_levels: max_gi = 'moderate'

            meal_events.append({
                "timestamp": timestamp,
                "date": timestamp.normalize().date(),
                "meal_type": 'snack' if meal_type == 'late_snack' else meal_type,
                "food_items": str(list(chosen_foods)),
                "portion_estimates": str([np.random.randint(80, 250) for _ in chosen_foods]),
                "gi_level": max_gi,
                "gi_multiplier": gi_map[max_gi]
            })

    meal_events = pd.DataFrame(meal_events).sort_values("timestamp").reset_index(drop=True)
    
    # 4. Sequential Simulation
    cgm_data.iloc[0, cgm_data.columns.get_loc('glucose_level')] = 110.0 # Initial glucose
    next_meal_idx = 0
    
    for i in range(1, len(cgm_data)):
        current_time = cgm_data.index[i]
        
        # Check if a meal is starting at this step
        if next_meal_idx < len(meal_events) and current_time >= meal_events.iloc[next_meal_idx]['timestamp']:
            meal = meal_events.iloc[next_meal_idx]
            meal_time_idx = i
            
            # --- Dynamic Baseline Calculation ---
            # 1. Rolling median of last 30 mins
            window_30min_idx = max(0, i - 6) # 6*5min = 30min
            baseline_glucose = np.nanmedian(cgm_data.iloc[window_30min_idx:i]['glucose_level'])
            
            # 2. Sleep quality adjustment
            day_of_meal = meal['timestamp'].normalize()
            sleep_quality = sleep_df[sleep_df['date'] == day_of_meal.date()]['sleep_quality'].iloc[0]
            if sleep_quality == 'poor':
                baseline_glucose += np.random.uniform(0, 10)
                
            # 3. Recent high-intensity activity adjustment
            recent_high_intensity = activity_df[
                (activity_df['intensity'] == 'high') &
                (activity_df['timestamp_start'] < meal['timestamp']) &
                (activity_df['timestamp_start'] > meal['timestamp'] - pd.Timedelta(hours=1))
            ]
            if not recent_high_intensity.empty:
                baseline_glucose -= 3
            
            meal_events.loc[next_meal_idx, 'baseline_glucose'] = round(baseline_glucose, 2)
            
            # --- Meal Impulse Calculation (Log-Normal) ---
            # Tuned to better match a non-insulin dependent T2D patient profile.
            GI_PEAK = {'low': 44, 'moderate': 60, 'high': 72}         # mg/dL peaks
            IMP_DUR = {'low': 24, 'moderate': 30, 'high': 36}        # 2h / 2.5h / 3h

            amp = GI_PEAK[meal['gi_level']]
            imp_dur = IMP_DUR[meal['gi_level']]

            # Log-normal params: s=sigma, scale=exp(mu). We target a mean of 35min.
            # mean = scale * exp(s^2/2) -> scale = mean / exp(s^2/2)
            s, mean_target = 0.4, 35
            scale = mean_target / np.exp(s**2 / 2)
            
            time_since_meal = (cgm_data.index[meal_time_idx:] - meal['timestamp']).total_seconds() / 60
            
            # Generate impulse for the next 2-3 hours
            impulse = lognorm.pdf(time_since_meal[:imp_dur], s=s, scale=scale, loc=0)
            
            # Normalize and scale impulse, then compute deltas
            normalized_impulse = (impulse / np.max(impulse)) * amp
            impulse_deltas = pd.Series(normalized_impulse).diff().fillna(0).values
            
            # Add to the main meal impulse curve
            end_idx = meal_time_idx + len(impulse_deltas)
            meal_impulse_delta[meal_time_idx:end_idx] += impulse_deltas
            
            next_meal_idx += 1
            
        # --- Update Glucose Level ---
        previous_glucose = cgm_data.iloc[i-1]['glucose_level']
        ar1_noise = np.random.normal(+0.02, 1.5)   # small mean drop
        
        cgm_data.iloc[i, cgm_data.columns.get_loc('glucose_level')] = (
            previous_glucose + 
            ar1_noise + 
            circadian_effect_delta[i] + 
            activity_effect_delta[i] +
            meal_impulse_delta[i] # Use the pre-calculated delta
        )
        
    # 5. Post-processing and Quality Flags
    
    # Sensor ID and warm-up gaps
    days_since_start = (cgm_data.index.normalize() - cgm_data.index[0].normalize()).days
    batch_idx = days_since_start // 14
    cgm_data['sensor_id'] = 'CGM-' + (batch_idx + 1).astype(str).str.zfill(3)
    
    # Set first 60 min of each new sensor to NaN
    for sensor in cgm_data['sensor_id'].unique():
        sensor_start_idx = cgm_data[cgm_data['sensor_id'] == sensor].index[0]
        warmup_end_time = sensor_start_idx + pd.Timedelta(minutes=60)
        warmup_mask = (cgm_data.index >= sensor_start_idx) & (cgm_data.index < warmup_end_time)
        cgm_data.loc[warmup_mask, 'glucose_level'] = np.nan
        cgm_data.loc[warmup_mask, 'signal_quality_flag'] = 'poor'

    # Clamp glucose floor to 60 mg/dL, values below become NaN
    low_glucose_mask = cgm_data['glucose_level'] < 65 # ↑ from 55 mg/dL
    cgm_data.loc[low_glucose_mask, 'signal_quality_flag'] = 'poor'
    cgm_data.loc[low_glucose_mask, 'glucose_level'] = np.nan
    cgm_data['glucose_level'] = cgm_data['glucose_level'].clip(upper=400) # Keep upper clip

    # Signal quality simulation
    quality_sample = np.random.choice(['good', 'fair', 'poor'], len(cgm_data), p=[0.90, 0.07, 0.03])
    # Apply where not already set by warm-up or low glucose
    cgm_data['signal_quality_flag'] = cgm_data['signal_quality_flag'].fillna(pd.Series(quality_sample, index=cgm_data.index))
    
    # Add extra noise for 'poor' quality signals
    poor_quality_mask = (cgm_data['signal_quality_flag'] == 'poor') & (cgm_data['glucose_level'].notna())
    cgm_data.loc[poor_quality_mask, 'glucose_level'] += np.random.uniform(-10, 10, size=poor_quality_mask.sum())

    # Final trend calculation
    cgm_data['glucose_trend'] = cgm_data['glucose_level'].diff().apply(lambda x: '↑' if x > 1 else ('↓' if x < -1 else '→'))
    
    # Join back other meal features for context
    final_meal_df = meal_events.merge(sleep_df[['date', 'sleep_quality']], on='date', how='left')
    final_meal_df = final_meal_df.drop(columns=['gi_multiplier'])
    
    return cgm_data.reset_index().rename(columns={'index': 'timestamp'}), final_meal_df

def compute_meal_targets(meal_df, cgm_df):
    """Computes postprandial metrics for each meal."""
    targets = []
    cgm_df_indexed = cgm_df.set_index('timestamp')

    for _, meal in meal_df.iterrows():
        t0 = meal['timestamp']
        baseline = meal['baseline_glucose']
        
        # Handle cases where meal occurs during NaN period
        if pd.isna(baseline):
            targets.append({
                'glucose_at_t+30min': np.nan, 'glucose_at_t+60min': np.nan,
                'AUC_postprandial_2h': np.nan, 'return_to_baseline_time': np.nan,
                'next_hypo_risk': np.nan, 'next_hyper_risk': np.nan
            })
            continue

        # Get glucose values at specific post-meal times
        t_plus_30_idx = cgm_df_indexed.index.get_indexer([t0 + pd.Timedelta(minutes=30)], method='nearest')[0]
        t_plus_60_idx = cgm_df_indexed.index.get_indexer([t0 + pd.Timedelta(minutes=60)], method='nearest')[0]
        
        glucose_30min = cgm_df_indexed.iloc[t_plus_30_idx]['glucose_level']
        glucose_60min = cgm_df_indexed.iloc[t_plus_60_idx]['glucose_level']
        
        # Calculate AUC for 2 hours post-meal
        post_meal_window = cgm_df_indexed[t0 : t0 + pd.Timedelta(hours=2)]
        auc_values = post_meal_window['glucose_level'].fillna(baseline) - baseline
        auc_values[auc_values < 0] = 0 # Only count area above baseline
        time_diff_hours = (post_meal_window.index.to_series().diff().dt.total_seconds() / 3600).fillna(0)
        auc = np.sum(auc_values * time_diff_hours)
        
        # Calculate return to baseline time
        try:
            return_time_idx = post_meal_window[
                (post_meal_window['glucose_level'].notna()) &
                (post_meal_window['glucose_level'] <= baseline + 5)
            ].index[0]
            return_to_baseline_min = int((return_time_idx - t0).total_seconds() / 60)
        except IndexError:
            return_to_baseline_min = 240 # Cap if not returned within window

        # Hypo/Hyper risk flags
        next_hypo_risk = 1 if baseline < 80 and meal['gi_level'] != 'high' else 0
        next_hyper_risk = 1 if (not pd.isna(glucose_30min) and glucose_30min >= 180) or \
                               (not pd.isna(glucose_60min) and glucose_60min >= 180) else 0
        
        targets.append({
            'glucose_at_t+30min': round(glucose_30min, 2) if pd.notna(glucose_30min) else np.nan,
            'glucose_at_t+60min': round(glucose_60min, 2) if pd.notna(glucose_60min) else np.nan,
            'AUC_postprandial_2h': round(auc, 2),
            'return_to_baseline_time': return_to_baseline_min,
            'next_hypo_risk': next_hypo_risk,
            'next_hyper_risk': next_hyper_risk
        })

    return pd.concat([meal_df, pd.DataFrame(targets, index=meal_df.index)], axis=1)

def generate_cgm_aggregates(days, cgm_df):
    """Computes 14-day trailing aggregates from CGM data."""
    aggregates = []
    cgm_df_indexed = cgm_df.set_index('timestamp')
    
    # Start from day 14 to have enough data
    for day in days[13:]:
        start_14d = day - pd.Timedelta(days=14)
        window_14d = cgm_df_indexed[start_14d:day]['glucose_level'].dropna()
        
        if not window_14d.empty:
            mean_g = window_14d.mean()
            sd_g = window_14d.std()
            mage = window_14d.quantile(0.9) - window_14d.quantile(0.1)
            
            aggregates.append({
                "date": day.date(),
                "mean_glucose": round(mean_g, 2),
                "sd": round(sd_g, 2),
                "cv": round(sd_g / mean_g if mean_g > 0 else 0, 2),
                "gmi": round(3.31 + 0.02392 * mean_g, 2),
                "time_in_range_pct": round(
                    ((window_14d >= 70) & (window_14d <= 180)).mean() * 100, 2
                ),
                "mage": round(mage, 2),
                "hypo_flag": int((window_14d < 70).any()),
                "hyper_flag": int((window_14d > 180).any()),
            })
    return pd.DataFrame(aggregates)

def print_summary_stats(cgm_df):
    """
    Prints summary statistics of the generated CGM data.
    Target metrics for a tuned T2D profile:
    - Mean glucose: 150-160 mg/dL
    - Time-in-range (70-180 mg/dL): ≥ 60%
    - Time > 180 mg/dL: ≤ 35%
    - Time < 70 mg/dL: ≤ 2%
    """
    print("\n--- CGM Summary Statistics ---")
    valid_glucose = cgm_df['glucose_level'].dropna()
    
    if valid_glucose.empty:
        print("No valid glucose data to analyze.")
        return
        
    mean_glucose = valid_glucose.mean()
    sd_glucose = valid_glucose.std()
    
    time_below_70 = (valid_glucose < 70).mean() * 100
    time_in_range = ((valid_glucose >= 70) & (valid_glucose <= 180)).mean() * 100
    time_above_180 = (valid_glucose > 180).mean() * 100
    
    # Largest 1h rise/fall
    hourly_diff = valid_glucose.diff(12).dropna() # 12 * 5 min = 1 hour
    max_rise = hourly_diff.max()
    max_fall = hourly_diff.min()
    
    print(f"  - Mean Glucose: {mean_glucose:.2f} mg/dL")
    print(f"  - SD Glucose: {sd_glucose:.2f} mg/dL")
    print(f"  - Time < 70 mg/dL: {time_below_70:.1f}%")
    print(f"  - Time 70-180 mg/dL: {time_in_range:.1f}%")
    print(f"  - Time > 180 mg/dL: {time_above_180:.1f}%")
    print(f"  - Largest 1h Rise: {max_rise:.2f} mg/dL")
    print(f"  - Largest 1h Fall: {max_fall:.2f} mg/dL")

def main():
    """Main function to generate and save all data artifacts."""
    days = pd.date_range(start=START_DATE, periods=NUM_DAYS)

    print("Generating sleep logs...")
    sleep_logs = generate_sleep_logs(days)
    
    print("Generating activity logs...")
    activity_logs = generate_activity_logs(days)
    
    print("Simulating CGM stream and meal data (this may take a moment)...")
    cgm_stream, meal_events_initial = generate_cgm_and_meal_data(days, sleep_logs, activity_logs)
    
    print("Computing post-meal targets...")
    meal_events = compute_meal_targets(meal_events_initial, cgm_stream)
    
    print("Computing 14-day CGM aggregates...")
    cgm_aggregates = generate_cgm_aggregates(days, cgm_stream)

    # --- SAVE TO CSV ---
    if not cgm_aggregates.empty:
        cgm_aggregates = cgm_aggregates[[
            "date","mean_glucose","gmi","time_in_range_pct",
            "cv","sd","mage","hypo_flag","hyper_flag"
        ]]
    meal_events = meal_events.drop(columns=['gi_level'])
    meal_events.to_csv("meal_events.csv", index=False)
    cgm_stream.to_csv("cgm_stream.csv", index=False)
    sleep_logs.to_csv("sleep_logs.csv", index=False)
    activity_logs.to_csv("activity_logs.csv", index=False)
    if not cgm_aggregates.empty:
        cgm_aggregates.to_csv("cgm_aggregates.csv", index=False)

    print("\n--- Data Generation Complete ---")
    print(f"  - meal_events.csv: {len(meal_events)} rows")
    print(f"  - cgm_stream.csv: {len(cgm_stream)} rows")
    print(f"  - sleep_logs.csv: {len(sleep_logs)} rows")
    print(f"  - activity_logs.csv: {len(activity_logs)} rows")
    if not cgm_aggregates.empty:
        print(f"  - cgm_aggregates.csv: {len(cgm_aggregates)} rows")
    else:
        print("  - cgm_aggregates.csv: 0 rows")
    print("\nCSVs saved to current directory.")
    
    print_summary_stats(cgm_stream)

    print("\nSimulated user profile:")
    for k, v in USER_PROFILE.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
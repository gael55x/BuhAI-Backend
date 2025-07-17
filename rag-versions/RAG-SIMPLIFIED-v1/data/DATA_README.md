# Data Specification: Synthetic Diabetes Data

This document outlines the schema and generation rules for five CSV files that model key data points for diabetes management in a Filipino context.

---

### 1. `cgm_stream.csv`

A continuous glucose monitoring (CGM) stream with readings every 5 minutes.

-   **`timestamp`**: (ISO-8601 `YYYY-MM-DDTHH:MM:SSZ`, Asia/Manila) The exact time of the sensor reading.
-   **`glucose_level`**: (float, mg/dL) The measured glucose value, typically ranging from 40 to 400.
-   **`glucose_trend`**: (string) A categorical indicator of glucose direction: `↑` (rising), `↓` (falling), or `→` (stable).
-   **`signal_quality_flag`**: (string) The quality of the sensor signal, categorized as `good` (95% of readings), `fair` (4%), or `poor` (1%).
-   **`sensor_id`**: (string) A unique, constant identifier for the CGM sensor (e.g., `CGM-001`).

---

### 2. `cgm_aggregates.csv`

Daily summary statistics calculated from the preceding 14 days of CGM data.

-   **`date`**: (YYYY-MM-DD) The date for which the aggregates are calculated.
-   **`mean_glucose`**: (float, mg/dL) The average glucose level over the 14-day window.
-   **`gmi`**: (float, %) Glucose Management Indicator, an estimate of HbA1c, calculated as `3.31 + 0.02392 * mean_glucose`.
-   **`time_in_range_pct`**: (float, %) The percentage of time glucose levels were within the target range of 70-180 mg/dL.
-   **`cv`**: (float) The coefficient of variation (`sd` / `mean_glucose`), a measure of glycemic variability.
-   **`sd`**: (float, mg/dL) The standard deviation of glucose levels.
-   **`mage`**: (float, mg/dL) Mean Amplitude of Glycemic Excursions, a measure of glucose swings. Calculated as the difference between the 90th and 10th percentile of glucose readings in the window.
-   **`hypo_flag`**: (integer, 0 or 1) `1` if any glucose reading in the window was < 70 mg/dL, otherwise `0`.
-   **`hyper_flag`**: (integer, 0 or 1) `1` if any glucose reading in the window was > 180 mg/dL, otherwise `0`.

---

### 3. `meal_events.csv`

A log of all food and drink intake.

-   **`timestamp`**: (ISO-8601 `YYYY-MM-DDTHH:MM:SSZ`, Asia/Manila) The start time of the meal.
-   **`meal_type`**: (string) The type of meal: `breakfast`, `lunch`, `dinner`, or `snack`.
-   **`food_items`**: (string) A semicolon-separated list of food items in `"item|grams"` format.

---

### 4. `sleep_logs.csv`

A log of nightly sleep patterns.

-   **`sleep_start`**: (ISO-8601 `YYYY-MM-DDTHH:MM:SSZ`, Asia/Manila) The time sleep began.
-   **`sleep_end`**: (ISO-8601 `YYYY-MM-DDTHH:MM:SSZ`, Asia/Manila) The time sleep ended.
-   **`duration_h`**: (float) The total duration of sleep in hours.
-   **`num_wakeups`**: (integer) The number of times sleep was disrupted.
-   **`sleep_quality`**: (string) Overall quality of sleep, `good` or `poor`, based on duration and disruptions.
-   **`was_disrupted`**: (integer, 0 or 1) `1` if `num_wakeups` > 0, otherwise `0`.

---

### 5. `activity_logs.csv`

A log of physical activities performed.

-   **`activity_type`**: (string) The type of activity: `walk`, `commute`, `chores`, `gym`, or `none`.
-   **`timestamp_start`**: (ISO-8601 `YYYY-MM-DDTHH:MM:SSZ`, Asia/Manila) The start time of the activity.
-   **`duration_min`**: (integer) The duration of the activity in minutes.
-   **`intensity`**: (string) The intensity level of the activity: `low`, `medium`, or `high`.
-   **`steps`**: (integer) The estimated number of steps taken during the activity.
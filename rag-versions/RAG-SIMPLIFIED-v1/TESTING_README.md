# BuhAI API Testing Guide

This guide provides quick `curl` commands to test the primary endpoints of the BuhAI API. It assumes the server is already running (`python app.py`).

## 1. Health Status Check

Verify that the API is running and the core logic has been initialized successfully.

```bash
curl http://localhost:4000/api/v1/status
```
**Expected Success Response:**
```json
{
  "status": "ok",
  "message": "BuhAI Core Logic is running."
}
```
---

## 2. Get a Personalized Insight

This is the main endpoint. It simulates a user logging a meal and returns a predictive insight.

```bash
curl -X POST http://localhost:4000/api/v1/insight \
-H "Content-Type: application/json" \
-d '{
    "user_id": "demo_user_01",
    "event_type": "meal",
    "event_data": {
        "description": "2 cups of white rice and fried chicken"
    }
}'
```
**Expected Response Structure:**
```json
{
  "alert_level": "WARNING",
  "bisaya_message": "Ang 2 cups of white rice and fried chicken makapasaka sa imong sugar...",
  "predicted_glucose_30min": "185.34",
  "recommendation_template": "post_meal_walk_recommendation",
  "risk_score": 0.7,
  "timestamp": "...",
  "user_id": "demo_user_01"
}
```
**Note:** Check the Flask server console for a critical alert message if the predicted glucose is very high.

---

## 3. Check for Missed Meal Alert

This endpoint checks if a meal has been logged during the current mealtime window (e.g., lunch from 12 PM to 3 PM). The response will depend on the current time and the data in the database.

```bash
curl http://localhost:4000/api/v1/inactivity-check
```
**Expected Response (if an alert should be triggered):**
```json
{
    "should_alert": true,
    "reason": "No meal logged during current 'lunch' window.",
    "last_log_timestamp": "...",
    "check_timestamp": "..."
}
```
**Expected Response (if no alert is needed):**
```json
{
    "should_alert": false,
    "reason": "Meal already logged during current 'lunch' window.",
    "last_log_timestamp": "...",
    "check_timestamp": "..."
}
```

---

## 4. Get Daily Progress Report

This endpoint generates a summary of the user's progress by comparing yesterday's stats to the day before.

```bash
curl http://localhost:4000/api/v1/daily-summary
```
**Expected Response Structure:**
```json
{
    "summary_text": "Nindot! Nimubo ang oras nga nilapas sa target range...",
    "date": "YYYY-MM-DD",
    "time_in_range_pct": 60.0
}
``` 
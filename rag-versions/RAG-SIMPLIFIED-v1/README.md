# BuhAI: A Personalized Hybrid RAG-based Diabetes Assistant

BuhAI is an AI-powered assistant designed to help Filipinos, particularly Cebuanos, manage diabetes through personalized, real-time insights. It combines Continuous Glucose Monitor (CGM) data with user-logged meals, activity, and sleep to provide predictive alerts and actionable recommendations in conversational Bisaya.

This project is built with a **Hybrid Retrieval-Augmented Generation (RAG)** architecture. This advanced approach enhances a standard RAG system by fusing retrieved unstructured text (past events) with structured data (a real-time glucose forecast from a predictive model), leading to more accurate and contextually-aware insights.

## Core Features

- **Personalized Insights**: Generates real-time advice based on user's logged meals and predicts future glucose levels.
- **Smart Alerts**: Proactively warns users of potential hypoglycemia or hyperglycemia events.
- **Emergency Notifications**: Includes a stubbed hook to notify next-of-kin or emergency services during critical events.
- **Inactivity Monitoring**: Intelligently checks for missed meal logs during key mealtimes.
- **Progress Tracking**: Provides daily summaries to help users track their progress over time.
- **Bisaya Interface**: All user-facing communication is in simple, conversational Cebuano.

## Architecture Overview

The system is composed of several key components:
1.  **Data Ingestion**: Scripts to process and load raw user data (CSV files) into a structured SQLite database (`data/buhai.db`).
2.  **Vector Store**: A ChromaDB vector database that stores embeddings of user logs for fast, semantic retrieval of **unstructured text** about past events.
3.  **Core Logic Layer**: The central "brain" that orchestrates the data flow. It uses historical data and the retrieved context from the RAG retriever to make decisions.
4.  **LLM for Generation**: Uses a Gemini LLM to translate the decisions from the core logic into friendly, user-facing Bisaya messages.
5.  **Flask API**: Exposes all functionality through a RESTful API for the frontend to consume.

## Getting Started

Follow these steps to set up and run the backend server.

### 1. Prerequisites

- Python 3.9+
- An active Google Gemini API Key

### 2. Setup

First, clone the repository and navigate into the project directory.

```bash
git clone <repository_url>
cd RAG
```

Create a Python virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

The application requires a Google Gemini API key to function.

Copy the example environment file:
```bash
cp .env.example .env
```
Now, open the `.env` file and add your Gemini API Key:
```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### 4. Data Ingestion

Before running the application for the first time, you must populate the database and the vector store with the sample data.

Run the ingestion script:
```bash
python ingest/ingest_csv.py
```
This script will:
- Create and populate the `data/buhai.db` SQLite database.
- Create and populate the ChromaDB vector store in the `vector_store/` directory.

### 5. Running the Application

Once the setup and data ingestion are complete, you can start the Flask server:
```bash
python app.py
```
The server will start on `http://0.0.0.0:5000`. You should see output indicating that the BuhAI Core Logic has been initialized.

## Testing the API

You can test the API endpoints using `curl` or any API client like Postman.

### Health Status

Check if the API and core logic are running correctly.

```bash
curl http://localhost:5000/api/v1/status
```
Expected Response:
```json
{
  "status": "ok",
  "message": "BuhAI Core Logic is running."
}
```

### Get a Personalized Insight

Simulate a user logging a meal. This is the primary endpoint of the application.

```bash
curl -X POST http://localhost:5000/api/v1/insight \
-H "Content-Type: application/json" \
-d '{
    "user_id": "demo_user_01",
    "event_type": "meal",
    "event_data": {
        "description": "2 cups of white rice and fried chicken"
    }
}'
```
The response will vary based on the data but will look something like this:
```json
{
  "alert_level": "WARNING",
  "bisaya_message": "Ang 2 cups of white rice and fried chicken makapasaka sa imong sugar. Ang 10-minutong paglakaw karon makatabang og dako para dili kaayo musaka ang spike.",
  "predicted_glucose_30min": "185.34",
  "recommendation_template": "post_meal_walk_recommendation",
  "risk_score": 0.7,
  "timestamp": "2024-05-22T10:30:00.123Z",
  "user_id": "demo_user_01"
}
```
The `predicted_glucose_30min` is now an *estimate* based on historical data for similar meals, not a forecast from a predictive model.

### Check for Missed Meal Alert

This endpoint checks if a meal has been logged during the current mealtime window. The response depends on the current time you run the command.

For example, run this between 12 PM and 3 PM:
```bash
curl http://localhost:5000/api/v1/inactivity-check
```
Expected response if no meal was logged since 12 PM:
```json
{
    "should_alert": true,
    "reason": "No meal logged during current 'lunch' window.",
    "last_log_timestamp": "...",
    "check_timestamp": "..."
}
```

### Get Daily Progress Report

This endpoint generates a summary comparing yesterday's performance to the day before.

```bash
curl http://localhost:5000/api/v1/daily-summary
```
Expected response:
```json
{
    "summary_text": "Nindot! Nimubo ang oras nga nilapas sa target range gikan sa 45% ngadto sa 60%. Padayon!",
    "date": "2024-05-21",
    "time_in_range_pct": 60.0
}
``` 
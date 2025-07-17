# BuhAI RAG V2: API Testing Guide

This guide provides a structured approach to testing the BuhAI API. It includes environment setup, application execution, and a suite of `curl` commands to validate the RAG pipeline's functionality.

---

## 1. Environment Setup

It is **critical** that you run this application with **Python 3.12** or newer. The dependencies are not compatible with older versions.

First, ensure your virtual environment is set up and activated correctly.

```bash
# Navigate to the project root if you aren't there
cd /Users/gailleamolong/Documents/Documents/MWEHE/

# Activate the virtual environment
source .venv/bin/activate

# Your terminal prompt should now start with (.venv)
# Navigate into the correct directory for running the app
cd RAG-v2
```

Next, ensure all dependencies are installed from within the activated environment.

```bash
pip install -r requirements.txt
```

---

## 2. Running the Application

Testing requires two steps: preparing the database and running the API server.

### A. Prepare the Database

This script deletes any old data and populates the database and vector store from the sample CSV files. Run this from the `RAG-v2` directory.

```bash
python ingest/ingest_csv.py
```

**Expected Output:**
You should see a series of log messages indicating successful ingestion for CGM, meals, activities, and sleep, ending with:
```
INFO:__main__:All data ingested successfully.
INFO:__main__:Vector store now contains 162 documents.
INFO:__main__:Database session closed.
```

### B. Start the API Server

Run the Flask application on port 4000. For testing, it's best to run this in a **separate terminal window** so you can see live logs and error messages.

```bash
python app.py --port 4000
```

**Expected Output:**
The server will start, and you will see messages from Werkzeug, including a debugger PIN. The server is ready when it is listening for connections.

---

## 3. Test Cases

Execute these `curl` commands from a new terminal window (not the one where the server is running).

### Test 1: Health Check

**Purpose:** Verify that the API is running and responsive.
```bash
curl -s http://localhost:4000/api/v1/status | jq
```
**Expected Response:**
```json
{
  "message": "BuhAI Chat Logic is available.",
  "status": "ok"
}
```

### Test 2: Normal-Flow (High-Carb Meal)

**Purpose:** Test a standard meal logging event.
```bash
curl -s -XPOST http://localhost:4000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id":"demo_user",
          "msg":"Ni kaon ko ug 2 tasa puti nga kan-on ug 120 grams lechon baboy",
          "ts":"2025-07-13T19:05:00"}' | jq
```
**Verification Points:**
- `reply_bisaya`: Should be a coherent, helpful message about the meal.
- `predictions`: Should contain `pred_30m` and `pred_60m` with sensible (elevated) glucose values.
- `summary`: Should be `null`.

### Test 3: Edge-Case (Unusual Quantity)

**Purpose:** Test how the system handles extreme or nonsensical inputs.
```bash
curl -s -XPOST http://localhost:4000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id":"demo_user",
          "msg":"Ni kaon ko ug 100000 grams kan-on",
          "ts":"2025-07-13T19:10:00"}' | jq
```
**Verification Points:**
- The API should not crash.
- `reply_bisaya`: Should still be a coherent response.
- `predictions`: Values should be high but within a reasonable range for the model.

### Test 4: Daily Summary Request

**Purpose:** Test the system's ability to correctly identify the 'summary' intent and retrieve data.
```bash
curl -s -XPOST http://localhost:4000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id":"demo_user",
          "msg":"Pwede ko makakuha sa akong daily summary?",
          "ts":"2025-07-13T22:05:00"}' | jq
```
**Verification Points:**
- `reply_bisaya`: Should contain a summary of the day's stats (Avg, Min, Max glucose, meal count).
- `summary`: Should be a JSON object containing the daily statistics, not `null`.
- `predictions`: Should be `null`.

### Test 5: General Chat

**Purpose:** Test the fallback 'general_chat' intent.
```bash
curl -s -XPOST http://localhost:4000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id":"demo_user",
          "msg":"Kumusta ka?",
          "ts":"2025-07-13T10:00:00"}' | jq
```
**Verification Points:**
- `reply_bisaya`: Should be a generic, friendly greeting.
- `summary` and `predictions`: Should both be `null`.

### Test 6: Alert Dispatcher (Hypoglycemia)

**Purpose:** Verify that a low-sugar event triggers a critical alert.
```bash
curl -s -XPOST http://localhost:4000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id":"demo_user",
          "msg":"Nangurog ko ug gibugnaw. Unsa akong buhaton?",
          "ts":"2025-07-13T11:00:00"}' | jq
```
**Verification Points:**
- Check the **server logs** for an `ALERT DISPATCHER` notice.
- `reply_bisaya`: Should contain first-aid advice for hypoglycemia.
- `alert_level`: Should be `WARNING` or `CRITICAL`.
- `actions`: Should contain `notify_nok`. 
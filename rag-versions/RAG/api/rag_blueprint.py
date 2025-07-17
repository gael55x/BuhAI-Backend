from flask import Blueprint, request, jsonify
import logging
from rag.core_logic import BuhaiCoreLogic
from datetime import datetime, timedelta

# --- BLUEPRINT SETUP ---
rag_bp = Blueprint('rag_bp', __name__, url_prefix='/api/v1')

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INITIALIZE CORE LOGIC ---
# This is created once when the blueprint is registered
try:
    logger.info("Initializing BuhAI Core Logic for the API...")
    core_logic = BuhaiCoreLogic()
    logger.info("BuhAI Core Logic initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not initialize BuhAI Core Logic. API will not work. Error: {e}", exc_info=True)
    core_logic = None

# --- MEALTIME CONFIG ---
MEAL_WINDOWS = {
    "breakfast": (7, 10),  # 7:00 AM to 9:59 AM
    "lunch": (12, 15),     # 12:00 PM to 2:59 PM
    "dinner": (18, 21),    # 6:00 PM to 8:59 PM
}

# --- API ENDPOINTS ---
@rag_bp.route('/daily-summary', methods=['GET'])
def get_daily_summary():
    """
    Generates a daily progress report comparing yesterday to the day before.
    """
    if not core_logic or not core_logic.data_handler:
        return jsonify({"error": "Core logic or data handler is not available."}), 500
    
    try:
        yesterday = datetime.now().date() - timedelta(days=1)
        summaries = core_logic.data_handler.get_daily_summary(yesterday)

        if not summaries or not summaries.get('today'):
            return jsonify({"summary_text": "Wala pay igo nga data para sa summary karong adlawa."})

        today_summary = summaries['today']
        yesterday_summary = summaries.get('yesterday')

        # Construct the prompt for the LLM
        prompt = """
        You are BuhAI, a friendly and encouraging AI health assistant.
        Generate a short, one-sentence summary in conversational Bisaya about the user's progress.
        Base it on the change in 'time in range percentage'. A higher percentage is better.

        Today's Time-in-Range: {today_pct}%
        Yesterday's Time-in-Range: {yesterday_pct}%

        - If it improved, say something like "Nindot! Nimubo ang oras nga nilapas sa target range..."
        - If it worsened, say something encouraging like "Okay lang na, naa pay ugma para mubawi. Padayon lang..."
        - If it's about the same, say something like "Maayo! Na-maintain nimo imong levels..."
        """

        today_pct = today_summary.time_in_range_pct * 100 if today_summary.time_in_range_pct else 0
        yesterday_pct = yesterday_summary.time_in_range_pct * 100 if yesterday_summary and yesterday_summary.time_in_range_pct else today_pct # Avoid comparing to None

        final_prompt = prompt.format(
            today_pct=f"{today_pct:.0f}",
            yesterday_pct=f"{yesterday_pct:.0f}"
        )

        # Generate the Bisaya output
        summary_text = core_logic.llm.generate_raw(final_prompt)

        return jsonify({
            "summary_text": summary_text.strip(),
            "date": yesterday.isoformat(),
            "time_in_range_pct": today_pct
        })

    except Exception as e:
        logger.error(f"An error occurred during daily summary generation: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@rag_bp.route('/inactivity-check', methods=['GET'])
def check_inactivity():
    """
    Checks if the user has failed to log a meal during a designated meal time.
    Returns a boolean flag to trigger alerts on the frontend.
    """
    if not core_logic or not core_logic.data_handler:
        return jsonify({"error": "Core logic or data handler is not available."}), 500

    now = datetime.now()
    current_hour = now.hour

    # 1. Check if we are within any meal window
    current_window_name = None
    window_start_hour = None
    for name, (start, end) in MEAL_WINDOWS.items():
        if start <= current_hour < end:
            current_window_name = name
            window_start_hour = start
            break

    if not current_window_name:
        return jsonify({
            "should_alert": False,
            "reason": f"Not a designated meal time. Current hour: {current_hour}."
        })

    # 2. If in a meal window, check for the last meal log
    try:
        last_meal_time = core_logic.data_handler.get_last_meal_timestamp()

        # Define the start of the current meal window for today
        window_start_time = now.replace(hour=window_start_hour, minute=0, second=0, microsecond=0)

        # 3. Compare last meal time to the start of the window
        if last_meal_time is None or last_meal_time < window_start_time:
            reason = "No meal logged during current '{}' window.".format(current_window_name)
            if last_meal_time is None:
                reason = "No meal logs ever found in the database."
            
            logger.warning(f"Inactivity alert triggered! Reason: {reason}")
            return jsonify({
                "should_alert": True,
                "reason": reason,
                "last_log_timestamp": last_meal_time.isoformat() if last_meal_time else None,
                "check_timestamp": now.isoformat()
            })
        else:
            return jsonify({
                "should_alert": False,
                "reason": f"Meal already logged during current '{current_window_name}' window.",
                "last_log_timestamp": last_meal_time.isoformat(),
                "check_timestamp": now.isoformat()
            })

    except Exception as e:
        logger.error(f"An error occurred during inactivity check: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@rag_bp.route('/insight', methods=['POST'])
def get_insight():
    """
    Main endpoint to get a personalized insight for a user event.
    """
    if not core_logic:
        return jsonify({"error": "Core logic is not available."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    user_id = data.get('user_id')
    event_type = data.get('event_type')
    event_data = data.get('event_data')

    if not all([user_id, event_type, event_data]):
        return jsonify({"error": "Missing required fields: user_id, event_type, event_data."}), 400

    try:
        logger.info(f"Received insight request for user '{user_id}' with event '{event_type}'")
        insight_response = core_logic.get_insights(user_id, event_type, event_data)
        
        if "error" in insight_response:
             return jsonify(insight_response), 500
             
        return jsonify(insight_response), 200

    except Exception as e:
        logger.error(f"An unexpected error occurred while generating insight: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@rag_bp.route('/status', methods=['GET'])
def get_status():
    """
    Simple health check endpoint for the API.
    Also can be expanded to check for user inactivity.
    """
    if core_logic:
        return jsonify({"status": "ok", "message": "BuhAI Core Logic is running."})
    else:
        return jsonify({"status": "error", "message": "BuhAI Core Logic failed to initialize."}), 500 
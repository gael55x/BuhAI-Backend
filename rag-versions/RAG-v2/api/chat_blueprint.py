from flask import Blueprint, request, jsonify, g
import logging
from rag.chat_logic import ChatLogic
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os

# --- BLUEPRINT SETUP ---
chat_bp = Blueprint('chat_bp', __name__, url_prefix='/api/v1')

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DATABASE CONNECTION ---
def get_db():
    if 'db' not in g:
        db_path = os.environ.get('DATABASE_URL', 'data/buhai.db')
        engine = create_engine(db_path)
        g.db = sessionmaker(bind=engine)()
    return g.db

@chat_bp.teardown_app_request
def teardown_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# --- INITIALIZE CORE LOGIC ---
def get_chat_logic():
    if 'chat_logic' not in g:
        g.chat_logic = ChatLogic(db_session=get_db())
    return g.chat_logic

# --- API ENDPOINTS ---
@chat_bp.route('/chat', methods=['POST'])
def handle_chat():
    """
    Main endpoint for the two-shot Gemini chat flow.
    """
    chat_logic = get_chat_logic()
    if not chat_logic:
        return jsonify({"error": "Chat logic is not available."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    user_id = data.get('user_id')
    message = data.get('msg')
    timestamp = data.get('ts', datetime.now().isoformat())

    if not all([user_id, message, timestamp]):
        return jsonify({"error": "Missing required fields: user_id, msg, ts."}), 400

    try:
        logger.info(f"Received chat from user '{user_id}'")
        response = chat_logic.process_chat(user_id, message, timestamp)
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"An unexpected error occurred during chat processing: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@chat_bp.route('/status', methods=['GET'])
def get_status():
    """
    Simple health check endpoint for the API.
    """
    # This doesn't need the full logic, just check if the code can be reached
    return jsonify({"status": "ok", "message": "BuhAI Chat Logic is available."}) 
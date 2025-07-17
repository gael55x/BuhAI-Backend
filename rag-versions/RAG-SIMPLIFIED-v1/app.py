from flask import Flask
from db.models import init_db
from api.chat_blueprint import chat_bp

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)

    # --- Database Initialization ---
    # It's good practice to initialize the DB with the app
    with app.app_context():
        init_db()

    # --- Register Blueprints ---
    # This is where you connect your modular API endpoints
    app.register_blueprint(chat_bp)

    @app.route("/")
    def index():
        return "BuhAI Chat API is running. Use the /api/v1/chat endpoint."

    return app

if __name__ == '__main__':
    app = create_app()
    # For development, you can run it like this.
    # For production, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(host='0.0.0.0', port=4000, debug=True) 
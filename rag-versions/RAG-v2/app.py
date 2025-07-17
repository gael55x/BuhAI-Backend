from flask import Flask
from db.models import init_db
from api.chat_blueprint import chat_bp
import os

def create_app(db_path=None):
    """Creates and configures the Flask application."""
    app = Flask(__name__)

    # --- Database Initialization ---
    if db_path is None:
        db_path = os.environ.get('DATABASE_URL', 'data/buhai.db')

    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    # Set the database URI for the app
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    with app.app_context():
        init_db(db_path)

    # --- Register Blueprints ---
    app.register_blueprint(chat_bp)

    @app.route("/")
    def index():
        return "BuhAI Chat API is running. Use the /api/v1/chat endpoint."

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=4000, debug=True) 
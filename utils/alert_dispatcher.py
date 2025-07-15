import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def notify(user_id: str, reason: str, message: str):
        logger.critical(f". User ID: {user_id}")
        logger.critical(f". Reason: {reason}")
        logger.critical(f". Message: {message}")
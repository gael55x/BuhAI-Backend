import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def notify(user_id: str, reason: str, message: str):
    """
    STUB: Sends a notification to the user's emergency contact.
    In a real system, this would trigger an SMS, push notification, or webhook.
    """
    logger.critical("--- ALERT DISPATCHER: NOTIFY ---")
    logger.critical(f"  User ID: {user_id}")
    logger.critical(f"  Reason: {reason}")
    logger.critical(f"  Message: {message}")
    logger.critical("--- END ALERT ---")

# Example of how it might be called:
# if __name__ == '__main__':
#     notify(
#         user_id="u1", 
#         reason="TRIGGER_ALERT_NOK", 
#         message="Critical hypoglycemia predicted. Please check on the user immediately."
#     ) 
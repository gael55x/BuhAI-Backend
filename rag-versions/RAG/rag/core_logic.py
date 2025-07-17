import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model

from rag.retriever import VectorRetriever
from rag.llm import GeminiLLM
from rag.data_handler import DataHandler, is_high_gi
from datetime import datetime

# --- CONFIGURATION ---
MODELS_DIR = Path("model/models")
SCALER_PATH = MODELS_DIR / "scaler_mv.pkl"
MODEL_PATH = MODELS_DIR / "lstm_mv_30.h5" # Using the 30-minute model
N_PAST_READINGS = 18 # 90 minutes of history (18 readings * 5 min/reading)
GLUCOSE_LEVEL_HYPO = 70
GLUCOSE_LEVEL_HIGH = 180
GLUCOSE_LEVEL_CRITICAL = 250


# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BuhaiCoreLogic:
    def __init__(self):
        """Initializes the core logic components."""
        logger.info("Initializing BuhAI Core Logic...")
        try:
            self.data_handler = DataHandler()
            self.retriever = VectorRetriever()
            self.llm = GeminiLLM()
            # The compile=False argument tells Keras to skip loading the optimizer and loss functions,
            # which are not needed for inference and can cause loading errors.
            self.model = load_model(MODEL_PATH, compile=False)
            self.scalers = joblib.load(SCALER_PATH)
            logger.info("Successfully loaded DataHandler, Retriever, LLM, ML model and scalers.")
        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}", exc_info=True)
            # Set components to None to prevent usage
            self.data_handler = None
            self.retriever = None
            self.llm = None
            self.model = None
            self.scalers = None

    def _run_rule_engine(self, predicted_glucose, retrieved_context, event_data):
        """
        Applies a set of rules to determine risk and a recommendation template.
        """
        # Default values
        risk_score = 0.2  # Start with a low base risk
        template = "general_insight"
        template_data = {}

        if predicted_glucose is not None:
            if predicted_glucose < GLUCOSE_LEVEL_HYPO:
                risk_score = 0.9
                template = "predicted_hypoglycemia"
                template_data = {"predicted_value": f"{predicted_glucose:.0f}"}
            
            elif predicted_glucose > GLUCOSE_LEVEL_CRITICAL:
                risk_score = 0.95
                template = "predicted_critical_hyperglycemia"
                template_data = {"predicted_value": f"{predicted_glucose:.0f}"}

            elif predicted_glucose > GLUCOSE_LEVEL_HIGH:
                risk_score = 0.7
                template = "predicted_hyperglycemia"
                template_data = {"predicted_value": f"{predicted_glucose:.0f}"}
                # Check if it was a high GI meal to suggest a walk
                if is_high_gi(event_data.get("description")):
                    template = "post_meal_walk_recommendation"

        # Check retrieved context for similar past events
        if not retrieved_context:
             return {"risk_score": risk_score, "template": template, "template_data": template_data}

        # Check for past high spikes with similar meals
        for context in retrieved_context:
            # A simple heuristic: if a past event text contains "hyper" or "high"
            if "hyper" in context.get("text", "") or "high" in context.get("text", ""):
                 # If we already have a high prediction, make the advice more specific
                if template == "predicted_hyperglycemia" or template == "post_meal_walk_recommendation":
                    template = "contextual_hyper_warning"
                    template_data["past_event"] = context.get("text")
                    risk_score = max(risk_score, 0.75) # Increase risk score
                break # Stop after finding one relevant past event
        
        return {"risk_score": risk_score, "template": template, "template_data": template_data}

    def _notify_emergency(self, payload: dict):
        """
        Stub function to simulate notifying emergency contacts.
        In a real system, this would trigger a webhook, SMS, or call.
        """
        logger.critical("--- EMERGENCY ALERT TRIGGERED ---")
        logger.critical(f"User: {payload.get('user_id')}")
        logger.critical(f"Reason: {payload.get('recommendation_template')}")
        logger.critical(f"Message: {payload.get('bisaya_message')}")
        logger.critical("--- NOTIFYING NEXT-OF-KIN / BARANGAY HEALTH WORKER ---")

    def _generate_bisaya_output(self, rule_output, event_data, retrieved_context):
        """Generates the final Bisaya output using a template-based prompt."""
        template = rule_output['template']
        template_data = rule_output['template_data']
        
        base_prompt = """
        You are BuhAI, a friendly and caring AI health assistant for Filipinos with diabetes.
        Your goal is to provide a clear, encouraging, and actionable message in simple, conversational Bisaya (Cebuano).
        Do not use deep or overly formal words. Keep it concise.
        
        Based on the following information, generate the appropriate response.
        """

        templates = {
            "predicted_hypoglycemia": f"""
                Instruction: Warn the user about a predicted low blood sugar event.
                Details: Predicted glucose is {template_data.get('predicted_value')} mg/dL.
                Example output: "EMERGENCY: Gina-estimate nga mu-us-os imong sugar sa {template_data.get('predicted_value')} mg/dL. Kaon og 15g nga asukal o pag-inom og juice karon dayon."
            """,
            "predicted_critical_hyperglycemia": f"""
                Instruction: Warn the user about a critically high blood sugar prediction.
                Details: Predicted glucose is {template_data.get('predicted_value')} mg/dL.
                Example output: "EMERGENCY: Nagpadulong ka sa {template_data.get('predicted_value')}+ mg/dL! Panahon na para mag-aksyon. Susiha imong insulin o pag-kontak sa doktor."
            """,
            "post_meal_walk_recommendation": f"""
                Instruction: The user ate a high-sugar meal and their glucose is predicted to be high. Recommend a short walk.
                Details: Predicted glucose is {template_data.get('predicted_value')} mg/dL after eating "{event_data.get('description')}".
                Example output: "Ang {event_data.get('description')} makapasaka sa imong sugar. Ang 10-minutong paglakaw karon makatabang og dako para dili kaayo musaka ang spike."
            """,
            "contextual_hyper_warning": f"""
                Instruction: Warn the user about a high prediction, reinforcing it with a similar past event.
                Details: Predicted glucose is high. A past event is: "{template_data.get('past_event')}". The current meal is "{event_data.get('description')}".
                Example output: "Pag-amping, Nanay/Tatay. Katong niaging kaon nimo nga parehas ana, misaka imong glucose. Suwayi og inom og daghang tubig."
            """,
            "general_insight": f"""
                Instruction: Provide a general, positive, or neutral insight.
                Details: The current meal is "{event_data.get('description')}". Retrieved context: {[c['text'] for c in retrieved_context]}
                Example output: "Salamat sa pag-log sa imong pagkaon. Importante ni para ma-monitor nato imong health. Padayon sa maayong buhat!"
            """
        }
        
        prompt_instruction = templates.get(template, templates['general_insight'])
        final_prompt = base_prompt + prompt_instruction
        
        return self.llm.generate_raw(final_prompt)

    def get_insights(self, user_id: str, event_type: str, event_data: dict):
        """
        The main orchestration method to generate insights for a user event.

        Args:
            user_id (str): The ID of the user.
            event_type (str): The type of event (e.g., 'meal', 'activity').
            event_data (dict): Data specific to the event (e.g., {'description': 'rice and chicken'}).

        Returns:
            dict: A dictionary containing the generated insights and recommendations.
        """
        if not all([self.data_handler, self.retriever, self.llm, self.model, self.scalers]):
            return {"error": "Core logic components are not initialized."}

        logger.info(f"Processing insight for user '{user_id}' and event '{event_type}'")
        
        # 1. Fetch recent user data from the database
        now = datetime.now()
        features_df = self.data_handler.get_prediction_features(end_time=now, window_minutes=N_PAST_READINGS * 5)
        
        predicted_glucose = None # Default to None
        if features_df.empty or len(features_df) < N_PAST_READINGS:
            logger.warning("Not enough data for LSTM prediction. Falling back to historical meal-based estimation.")
            # Fallback to a rule-based estimation if event is a meal
            if event_type == 'meal' and 'description' in event_data:
                predicted_glucose = self.data_handler.get_historical_meal_estimate(event_data['description'])
        else:
            # 2. Prepare feature vector for the ML model
            # Ensure we have exactly N_PAST_READINGS for the model
            input_features = features_df.tail(N_PAST_READINGS)
            
            # Scale the features
            scaled_features = np.zeros_like(input_features.values)
            for i, col in enumerate(input_features.columns):
                scaler = self.scalers[col]
                scaled_features[:, i] = scaler.transform(input_features[[col]]).flatten()
            
            # Reshape for the model [samples, timesteps, features]
            model_input = np.expand_dims(scaled_features, axis=0)

            # 3. Get numeric prediction from the ML model
            predicted_scaled = self.model.predict(model_input)
            
            # Inverse transform the prediction
            glucose_scaler = self.scalers['glucose_level']
            predicted_glucose = glucose_scaler.inverse_transform(predicted_scaled)[0][0]
            logger.info(f"Predicted glucose in 30 mins: {predicted_glucose:.2f} mg/dL")

        # 4. Get retrieved context from the vector store
        query_text = f"User event: {event_type}. Details: {event_data.get('description', 'N/A')}"
        retrieved_context = self.retriever.query(query_text, k=3)
        
        # 5. Implement Rule Engine and Risk Assessor
        rule_engine_output = self._run_rule_engine(predicted_glucose, retrieved_context, event_data)
        
        # 6. Select recommendation template (done inside rule engine)
        
        # 7. Generate final Bisaya output using LLM
        bisaya_message = self._generate_bisaya_output(rule_engine_output, event_data, retrieved_context)
        
        # 8. Format and return the final output
        final_output = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "predicted_glucose_30min": f"{predicted_glucose:.2f}" if predicted_glucose is not None else None,
            "risk_score": rule_engine_output['risk_score'],
            "recommendation_template": rule_engine_output['template'],
            "bisaya_message": bisaya_message.strip(),
            "alert_level": "NONE"
        }
        
        if rule_engine_output['risk_score'] >= 0.9:
            final_output['alert_level'] = "CRITICAL"
        elif rule_engine_output['risk_score'] >= 0.7:
            final_output['alert_level'] = "WARNING"
            
        # If the alert is critical, call the emergency hook
        if final_output['alert_level'] == "CRITICAL":
            self._notify_emergency(final_output)

        return final_output

# --- Example Usage ---
if __name__ == '__main__':
    core_logic = BuhaiCoreLogic()
    if core_logic.model:
        # Example for a user logging a meal
        user = "test_user_123"
        event = "meal"
        data = {"description": "1 cup of white rice and fried chicken"}
        
        insights = core_logic.get_insights(user, event, data)
        print("\n--- Core Logic Output ---")
        print(insights)
        print("-------------------------\n") 
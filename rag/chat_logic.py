import logging
from datetime import datetime, date
import json
import random
import re

from rag.data_handler import DataHandler
from rag.retriever import VectorRetriever
from rag.llm import GeminiLLM
from utils import alert_dispatcher

MEAL_WINDOWS = {
    "breakfast": (7, 10),
    "lunch": (12, 15),
    "dinner": (18, 21),
}
SIMILARITY_THRESHOLD = 0.85  # For vector search
GLUCOSE_LEVELS = {
    "HYPO": 70,
    "HIGH": 180,
    "CRITICAL": 250,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatLogic:
    def __init__(self, db_session=None):
        self.data_handler = DataHandler(db_session=db_session)
        self.retriever = VectorRetriever()
        self.llm = GeminiLLM()

    def _estimate_glucose_fallback(self, food_description: str) -> dict:
        is_high_gi = any(keyword in food_description.lower() for keyword in ["rice", "bread", "soda", "cake", "sweet"])
        base_rise = 50 if is_high_gi else 30
        pred_30m = 120 + base_rise + random.uniform(-5, 5)
        pred_60m = 110 + (base_rise / 2) + random.uniform(-5, 5)
        return {"pred_30m": pred_30m, "pred_60m": pred_60m, "delta": base_rise}

    def _estimate_glucose_with_llm(self, food_description: str) -> dict:
        # Uses Gemini to estimate glucose rise based on a meal description.

        prompt = f"""
        Analyze the following meal description from a user in the Philippines.
        Your task is to estimate the impact on their blood glucose.

        Meal Description: "{food_description}"

        Consider common Filipino foods and portion sizes.
        - "kan-on" or "rice" is typically white rice. A "tasa" or "cup" is about 180g.
        - Lechon baboy (pork) has minimal carbs.
        - Be realistic. A person cannot eat 100,000 grams of rice. Cap the estimated impact for absurd quantities.

        Respond with ONLY a JSON object containing your estimate. The format should be:
        {{
          "estimated_glycemic_impact": <integer>
        }}
        
        The "estimated_glycemic_impact" should be an integer representing the expected rise in blood glucose (mg/dL) 30 minutes after the meal. For a normal, balanced meal, this might be between 40-80. For a very high-carb meal, it could be 100-150. For an absurdly large meal like "100,000 grams of rice", use a capped, high value like 250.
        """
        
        try:
            response_text = self.llm.generate_raw(prompt)
            json_str = response_text.strip().replace("```json", "").replace("```", "")
            data = json.loads(json_str)
            base_rise = int(data.get("estimated_glycemic_impact", 50))
            
            pred_30m = 120 + base_rise + random.uniform(-5, 5)
            pred_60m = 110 + (base_rise * 0.75) + random.uniform(-5, 5)
            
            logger.info(f"LLM estimated glucose impact: delta={base_rise}")
            return {"pred_30m": pred_30m, "pred_60m": pred_60m, "delta": base_rise}

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response for glucose estimation: {e}. Falling back to simple estimator.")
            return None

    def _pattern_estimate(self, message: str) -> dict:
        # Performs vector search for food items and falls back to a GI estimator.
        llm_estimate = self._estimate_glucose_with_llm(message)
        if llm_estimate:
            llm_estimate["source"] = "llm_estimator"
            return llm_estimate

        neighbors = self.retriever.query(message, k=1)
        if neighbors and neighbors[0]['score'] >= SIMILARITY_THRESHOLD:
            logger.info(f"Found similar meal: {neighbors[0]['text']}")
            delta = random.uniform(40, 90)
            return {
                "pred_30m": 120 + delta,
                "pred_60m": 110 + delta / 2,
                "delta": delta,
                "source": "historical"
            }
        else:
            logger.info("No similar meal found, using GI fallback estimator.")
            estimate = self._estimate_glucose_fallback(message)
            estimate["source"] = "gi_estimator"
            return estimate

    def _assemble_context(self, user_id: str, ts: datetime) -> dict:
        """Gathers all necessary context for the LLM prompts, excluding meal-specific predictions."""
        context = {
            'cgm_readings': self.data_handler.get_last_cgm_readings(n=3),
            'stats': self.data_handler.get_daily_and_weekly_stats(ts.date()),
            'insight_data': self.data_handler.get_contextual_data_for_insights(ts.date()),
            'chat_history': self.data_handler.get_chat_history(user_id, n=4),
            'missed_meal': not any(start <= ts.hour < end for start, end in MEAL_WINDOWS.values())
        }
        return context

    def _format_summary_for_llm(self, stats: dict) -> str:
        """Formats the stats dictionary into a human-readable string for the LLM."""
        if not stats or not stats.get("daily"):
            return "No summary data available for you at the moment."

        daily = stats["daily"]
        weekly = stats.get("weekly", {})

        # Use the date from the stats, which is now reliably calculated from the stream
        try:
            daily_date_str = daily.get('date', '').split('T')[0]
            daily_date = datetime.strptime(daily_date_str, '%Y-%m-%d')
            date_formatted = daily_date.strftime('%B %d, %Y')
        except (ValueError, TypeError):
            date_formatted = "the requested day"

        formatted_string = f"**Daily Summary for {date_formatted}:**\\n"
        
        if daily.get("mean_glucose") is not None:
            formatted_string += f"- Average Glucose: {daily.get('mean_glucose', 'N/A'):.2f} mg/dL\\n"
            formatted_string += f"- Time in Range: {daily.get('time_in_range_pct', 'N/A'):.2f}%\\n"
            formatted_string += f"- GMI (Est. A1c): {daily.get('gmi', 'N/A'):.2f}\\n"
        else:
            # explicitly state that no CGM data is available
            # and that insights will be based on other logged data.
            formatted_string += "- No CGM data was recorded for this day. I will provide insights based on your logged meals, activities, or my own glucose estimations for your logged meals.\\n"

        if weekly and weekly.get("avg_glucose") is not None:
            formatted_string += f"\\n**Weekly Summary (ending {date_formatted}):**\\n"
            formatted_string += f"- Average Glucose: {weekly.get('avg_glucose', 'N/A'):.2f} mg/dL\\n"
            formatted_string += f"- Average Time in Range: {weekly.get('avg_time_in_range_pct', 'N/A'):.2f}%\\n"

        return formatted_string

    def _format_insight_data_for_llm(self, insight_data: dict) -> str:
        """Formats the insight data into a readable string for the LLM."""
        if not insight_data or not any(insight_data.values()):
            return "No recent meal, sleep, or activity data logged."

        lines = []
        if insight_data.get("recent_meals"):
            lines.append("- Recent Meals:")
            for meal in insight_data["recent_meals"]:
                ts = datetime.fromisoformat(meal['timestamp']).strftime('%b %d, %I:%M %p')
                lines.append(f"  - At {ts}: Ate '{meal['food_items']}'.")
        
        if insight_data.get("recent_sleep"):
            lines.append("- Recent Sleep:")
            for sleep in insight_data["recent_sleep"]:
                ts = datetime.fromisoformat(sleep['sleep_start']).strftime('%b %d')
                lines.append(f"  - Night of {ts}: Slept {sleep['duration_h']:.2f} hours (Quality: {sleep['sleep_quality']}).")

        if insight_data.get("recent_activity"):
            lines.append("- Recent Activity:")
            for activity in insight_data["recent_activity"]:
                ts = datetime.fromisoformat(activity['timestamp_start']).strftime('%b %d, %I:%M %p')
                lines.append(f"  - At {ts}: {activity['activity_type']} for {activity['duration_min']} minutes.")
        
        return "\\n".join(lines)

    def _execute_tasks(self, tasks: list, context: dict, user_id: str, message: str) -> dict:
        """Executes tasks returned by the Router LLM."""
        execution_results = {}
        first_aid_text = None
        actions = []

        if "db_log" in tasks:
            logger.info("TASK: Logging meal to database.")

        if "trigger_alert_nok" in tasks:
            actions.append("notify_nok")
            alert_dispatcher.notify(user_id, "TRIGGER_ALERT_NOK", "User requires attention.")
        
        if "need_activity_reco" in tasks and "glucose_prediction" in context:
            delta = context["glucose_prediction"].get("delta", 0)
            reco = "a 15-minute walk or some light chores."
            if delta < 40:
                reco = "a 5-minute walk."
            elif 40 <= delta < 80:
                reco = "a 10-minute brisk walk."
            execution_results["activity_recommendation"] = f"To manage the spike, I recommend {reco}"

        if "glucose_prediction" in context:
            pred_30m = context["glucose_prediction"]["pred_30m"]
            if pred_30m < GLUCOSE_LEVELS["HYPO"]:
                first_aid_text = "Eat 15g of sugar or drink juice now."
                if "notify_nok" not in actions: actions.append("notify_nok")
            elif pred_30m >= GLUCOSE_LEVELS["CRITICAL"]:
                first_aid_text = "Your sugar is critically high. Consider your insulin plan or contact a doctor."
                if "notify_nok" not in actions: actions.append("notify_nok")

        execution_results["first_aid"] = first_aid_text
        execution_results["actions"] = actions
        return execution_results

    def _format_chat_history_for_llm(self, chat_history: list) -> str:
        """Formats the chat history into a string for the LLM prompt."""
        if not chat_history:
            return "No previous conversation history."
        
        formatted_history = "\\n".join(
            [f"- {turn.actor}: {turn.message}" for turn in reversed(chat_history)]
        )
        return formatted_history

    def _get_router_response_with_llm(self, message: str, context: dict) -> dict:
        # Uses an LLM to determine intent and tasks.
        prompt = f"""
        You are a router. Analyze the user's message and context to determine the intent and necessary tasks.
        User message: "{message}"
        Context: {json.dumps(context, default=str)}
        
        Possible intents: 'log_meal', 'get_summary_day', 'get_summary_week', 'general_chat', 'question_about_previous_meal'.
        Possible tasks: 'db_log', 'trigger_alert_nok', 'need_activity_reco'.
        
        - If the user mentions eating, food, or drinks, the intent is 'log_meal'.
        - If the user asks what they ate, the intent is 'question_about_previous_meal'.
        - If the user asks for a "summary" or "report", the intent is 'get_summary_day'.
        - Otherwise, the intent is 'general_chat'.

        Respond with ONLY a JSON object: {{"intent": "...", "tasks": [...]}}
        If the intent is 'log_meal', always include "need_activity_reco" and "db_log" in tasks.
        """
        try:
            response_text = self.llm.generate_raw(prompt)
            json_str = response_text.strip().replace("```json", "").replace("```", "")
            router_response = json.loads(json_str)
            logger.info(f"Router LLM response: {router_response}")
            return router_response
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse router LLM response: {e}. Defaulting to general_chat.")
            return {"intent": "general_chat", "tasks": []}

    def process_chat(self, user_id: str, message: str, ts_str: str) -> dict:
        # 2 shot gemini flow
        ts = datetime.fromisoformat(ts_str)

        self.data_handler.add_chat_turn(user_id, 'user', message)

        context = self._assemble_context(user_id, ts)

        router_response = self._get_router_response_with_llm(message, context)
        
        if router_response.get("intent") == "log_meal":
            context['glucose_prediction'] = self._pattern_estimate(message)

        execution_results = self._execute_tasks(router_response.get("tasks", []), context, user_id, message)

        final_response = ""
        try:
            formatted_summary = ""
            if router_response['intent'].startswith('get_summary') and context.get('stats'):
                formatted_summary = self._format_summary_for_llm(context['stats'])

            formatted_chat_history = self._format_chat_history_for_llm(context.get('chat_history', []))
            
            formatted_insight_data = self._format_insight_data_for_llm(context.get("insight_data"))

            final_response_prompt = f"""
            You are BuhAI, a friendly and empathetic AI health assistant for diabetics.
            Your main goal is to respond to the user in clear, supportive English. Be concise, easy to understand, and encouraging.

            **Conversation History:**
            {formatted_chat_history}

            **Current User Message:**
            "{message}"

            **Your Task: Construct a helpful response by following these steps precisely.**

            **Step 1: Current Data & Context**
            *   **User's Requested Date:** {ts.strftime('%B %d, %Y')}
            *   **Daily/Weekly Summary:** {formatted_summary}
            *   **Recent Logs (Meals, Sleep, Activity):**
                {formatted_insight_data}
            *   **Glucose Prediction (if a meal was just logged):** {context.get('glucose_prediction')}
            *   **Activity Recommendation (if applicable):** {execution_results.get('activity_recommendation')}
            *   **First Aid Notice (CRITICAL):** {execution_results.get('first_aid')}

            **Step 2: Response Generation Rules**
            1.  **Language:** Respond in clear, supportive English.
            2.  **Tone:** Be empathetic, encouraging, and clear.
            3.  **Critical First Aid:** If `First Aid Notice` exists, it is the MOST IMPORTANT information. Start your response with it.
            4.  **Address the Query:** Directly address the user's question. If they ask for "insights" or a "summary," use the data provided.
            5.  **Handling No CGM Data:** If the summary says "No CGM data was recorded", you MUST state this clearly. Then, derive your main insight from the `Recent Logs` or `Glucose Prediction`. **DO NOT invent CGM data.**
            6.  **Generate ONE Key Insight:** Look at all the context. Find the most interesting or important pattern. This is your main point. Examples:
                - "I noticed that when you sleep longer, your average glucose seems to be lower the next day. That's great!"
                - "Based on my estimate for the meal you just logged, a short 10-minute walk could help manage the potential glucose spike."
                - "I see you had a high-intensity workout yesterday. That can sometimes affect blood sugar for hours afterward."
            7.  **BE SPECIFIC:** Tie your insight to specific data. Instead of "your sleep was better," say "you slept 8.5 hours."
            8.  **AVOID "You did not log...":** Instead of scolding the user for not logging, encourage them to log future events. For example: "To help me give you better predictions, try logging your next meal."
            9.  **Structure:** Your response should have a clear flow:
                - Friendly greeting ("Hello!" or "Hi there!")
                - Address the user's question (present summary or state no data).
                - Provide your single, key insight with a brief explanation.
                - Offer a supportive closing statement ("I'm here to help you manage your diabetes.")

            **Step 3: Write Your Response**
            Based on all the rules and data, generate your final response now. Limit to 50-70 words depending on the context.

            **Your Response:**
            """

            final_response = self.llm.generate_raw(final_response_prompt)
            
            self.data_handler.add_chat_turn(user_id, 'assistant', final_response)

        except Exception as e:
            logger.error(f"Error in Final Response Generation: {e}", exc_info=True)
            final_response = "I'm sorry, I'm experiencing a technical issue. Please try again in a moment."
            if self.data_handler:
                self.data_handler.add_chat_turn(user_id, 'assistant', final_response)
            
        api_response = {
            "reply_bisaya": final_response,  # Keep the key name for frontend compatibility
            "reply": final_response,  # Add new key for future use
            "alert_level": "WARNING" if execution_results.get("first_aid") else "NONE",
            "predictions": context.get("glucose_prediction"),
            "summary": context.get('stats') if router_response.get('intent', '').startswith('get_summary') else None,
            "actions": execution_results.get("actions", [])
        }

        return api_response 

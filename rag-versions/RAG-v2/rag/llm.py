import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiLLM:
    def __init__(self):
        """Initializes the Gemini LLM wrapper."""
        if GEMINI_API_KEY:
            logger.info("GEMINI_API_KEY found. Configuring Gemini LLM.")
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                # Using a cost-effective and fast model suitable for RAG
                self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
                logger.info("Gemini LLM initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to configure Gemini SDK: {e}", exc_info=True)
                self.model = None
        else:
            logger.warning("GEMINI_API_KEY not found. LLM will return dummy answers.")
            self.model = None

    def generate_raw(self, prompt: str) -> str:
        """
        Generates content directly from a provided prompt string.

        Args:
            prompt (str): The full prompt to send to the LLM.

        Returns:
            str: The generated text.
        """
        if not self.model:
            return "DUMMY_ANSWER: No API key found."
        
        logger.info("Generating answer with Gemini LLM from raw prompt...")
        try:
            generation_config = {"temperature": 0.7}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            logger.error(f"An error occurred during LLM generation: {e}", exc_info=True)
            return "Error: Could not generate an answer due to an API error."

    def generate(self, question: str, context: list) -> str:
        """
        Generates an answer based on the question and provided context.

        Args:
            question (str): The user's question.
            context (list): A list of context strings retrieved from the vector store.

        Returns:
            str: The generated answer.
        """
        if not self.model:
            return "DUMMY_ANSWER: No API key found."

        # Construct a prompt from the context and question
        context_str = "\n".join([f"- {item['text']}" for item in context])
        prompt = f"""
        You are a helpful assistant for a person with diabetes.
        Based on the following context about the user's recent health data, please answer the question.
        Provide a concise and direct answer. If the context does not provide enough information, state that clearly.

        Context:
        {context_str}

        Question: {question}

        Answer:
        """

        logger.info("Generating answer with Gemini LLM...")
        
        try:
            # Call the actual Gemini API
            generation_config = {"temperature": 0.7}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text

        except Exception as e:
            logger.error(f"An error occurred during LLM generation: {e}", exc_info=True)
            return "Error: Could not generate an answer due to an API error."

# --- Example Usage ---
if __name__ == '__main__':
    llm = GeminiLLM()

    # Example with a dummy context
    dummy_context = [
        {'text': 'User ate lechon kawali for breakfast on 2025-06-12.'},
        {'text': 'Daily glucose average on 2025-06-12 was 153.80 mg/dL.'}
    ]
    test_question = "What was my blood sugar after eating lechon?"

    answer = llm.generate(question=test_question, context=dummy_context)

    print(f"\nQuestion: {test_question}")
    print(f"Answer: {answer}\n")

    # Example with raw prompt
    raw_prompt_example = "Translate 'Hello, how are you?' to Cebuano."
    raw_answer = llm.generate_raw(raw_prompt_example)
    print(f"\nRaw Prompt: {raw_prompt_example}")
    print(f"Answer: {raw_answer}\n") 
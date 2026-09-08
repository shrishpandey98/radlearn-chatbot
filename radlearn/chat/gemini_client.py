"""
radlearn/chat/gemini_client.py
──────────────────────────────
Gemini client implementing the BaseLLMClient interface.
"""
import time
import logging
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded
from radlearn.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from radlearn.chat.llm_base import BaseLLMClient

logger = logging.getLogger(__name__)

# Configure globally (as per existing integration)
genai.configure(api_key=GOOGLE_API_KEY)

_generation_config = {
    "temperature": LLM_TEMPERATURE,
    "max_output_tokens": LLM_MAX_TOKENS,
}

class GeminiClient(BaseLLMClient):
    """
    Handles interactions with Google's Gemini models using robust quota handling.
    """
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)


    def generate_answer(self, system_instruction: str, user_prompt: str) -> str:
        """
        Calls Gemini to generate an answer with exponential backoff for rate limits.
        """
        model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            system_instruction=system_instruction,
            generation_config=_generation_config
        )
        
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = model.generate_content(user_prompt)
                return response.text
            except (ResourceExhausted, DeadlineExceeded) as e:
                # We catch ResourceExhausted (quota limits) and allow it to bubble up 
                # after retries are exhausted, or bubble up immediately if we choose.
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Gemini API rate limit/timeout. Retrying in {wait_time}s... (Attempt {attempt}/{self.max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Gemini API Error: {e}")
                raise e
                
        # If we exhausted all retries, raise the last ResourceExhausted error
        # so the engine can catch it and return "quota_exceeded"
        raise last_error or RuntimeError("Failed to generate answer.")

    def generate_stream(self, system_instruction: str, user_prompt: str):
        model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            system_instruction=system_instruction,
            generation_config=_generation_config
        )
        
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = model.generate_content(user_prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                return # Exit successfully
            except (ResourceExhausted, DeadlineExceeded) as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Gemini API rate limit/timeout during stream. Retrying in {wait_time}s... (Attempt {attempt}/{self.max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Gemini API Error in stream: {e}")
                raise e
                
        raise last_error or RuntimeError("Failed to generate stream.")

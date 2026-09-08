import os
from groq import Groq
from radlearn.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from radlearn.chat.llm_base import BaseLLMClient

class GroqClient(BaseLLMClient):
    """
    Implementation of the Groq API client using the official SDK.
    """
    def __init__(self):
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
            
        self.client = Groq(api_key=api_key)
        self.model_name = LLM_MODEL


    def generate_answer(self, system_instruction: str, user_prompt: str) -> str:
        """
        Generates an answer using Groq's chat completions API.
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=self.model_name,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Map standard Groq rate limit errors so the engine can catch them
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RuntimeError(f"ResourceExhausted: {str(e)}")
            raise e

    def generate_stream(self, system_instruction: str, user_prompt: str):
        """
        Generates an answer using Groq's chat completions API as a stream.
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=self.model_name,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RuntimeError(f"ResourceExhausted: {str(e)}")
            raise e

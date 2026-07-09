"""
radlearn/chat/openai_client.py
──────────────────────────────
OpenAI client stub implementing the BaseLLMClient interface.
"""
from radlearn.chat.llm_base import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    """
    Placeholder for future OpenAI integration (GPT-4o, etc).
    """
    def __init__(self):
        pass

    def generate_answer(self, system_instruction: str, user_prompt: str) -> str:
        raise NotImplementedError("OpenAI integration coming soon. See LLM abstraction layer.")

    def generate_stream(self, system_instruction: str, user_prompt: str):
        raise NotImplementedError("OpenAI streaming integration coming soon.")

"""
radlearn/chat/anthropic_client.py
─────────────────────────────────
Anthropic client stub implementing the BaseLLMClient interface.
"""
from radlearn.chat.llm_base import BaseLLMClient

class AnthropicClient(BaseLLMClient):
    """
    Placeholder for future Anthropic integration (Claude 3.5 Sonnet, etc).
    """
    def __init__(self):
        pass

    def generate_answer(self, system_instruction: str, user_prompt: str) -> str:
        raise NotImplementedError("Anthropic integration coming soon. See LLM abstraction layer.")

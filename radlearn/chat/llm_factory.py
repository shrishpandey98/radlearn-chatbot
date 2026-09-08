"""
radlearn/chat/llm_factory.py
────────────────────────────
Factory pattern to instantiate the appropriate LLM client based on config.
"""
from radlearn.config import LLM_PROVIDER
from radlearn.chat.llm_base import BaseLLMClient

def get_llm_client() -> BaseLLMClient:
    """
    Factory function to return the configured LLM client.
    """
    provider = LLM_PROVIDER.lower()


    if provider == "groq":
        from radlearn.chat.groq_client import GroqClient
        return GroqClient()
    elif provider == "gemini":
        from radlearn.chat.gemini_client import GeminiClient
        return GeminiClient()
    elif provider == "openai":
        from radlearn.chat.openai_client import OpenAIClient
        return OpenAIClient()
    elif provider == "anthropic":
        from radlearn.chat.anthropic_client import AnthropicClient
        return AnthropicClient()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

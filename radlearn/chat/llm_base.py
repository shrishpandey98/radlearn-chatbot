"""
radlearn/chat/llm_base.py
─────────────────────────
Abstract base class for LLM providers to ensure a consistent interface.
"""
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """
    Abstract interface for all LLM integrations (Gemini, OpenAI, Anthropic).
    """

    @abstractmethod
    def generate_answer(self, system_instruction: str, user_prompt: str) -> str:
        """
        Generates an answer from the LLM based on the system instructions and user prompt.
        
        Args:
            system_instruction: The core system prompt defining the LLM's persona/behavior.
            user_prompt: The augmented user prompt containing the question and retrieved context.
            
        Returns:
            The raw text string of the generated answer.
            
        Raises:
            Exception: Any provider-specific exception (e.g., ResourceExhausted for quota limits).
        """
        pass

    @abstractmethod
    def generate_stream(self, system_instruction: str, user_prompt: str):
        """
        Generates an answer from the LLM as a stream of strings.
        """
        pass

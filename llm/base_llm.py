"""Base class for any LLM used"""

from abc import ABC, abstractmethod
from typing import List


class BaseLLM(ABC):
    """Base class for any LLM used"""

    @abstractmethod
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response based on the query and context

        Args:
            query (str): User's question
            context (List[str]): Retrieved context documents

        Returns:
            str: Generated response
        """

    def __str__(self) -> str:
        """Return string representation of the model

        Returns:
            str: Model name and type
        """
        return f"{self.__class__.__name__}"

"""Base class for any embedding model used"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """
    Base Abstract class for any embedding model used
    """

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for the given texts

        Args:
            texts (List[str]): List of texts to get embeddings for

        Returns:
            List[List[float]]: List of embeddings
        """

    def __str__(self) -> str:
        """Return string representation of the model

        Returns:
            str: Model name and type
        """
        return f"{self.__class__.__name__}"

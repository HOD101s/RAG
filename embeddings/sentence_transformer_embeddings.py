"""Embedding model using Sentence Transformers"""

from typing import List

from sentence_transformers import SentenceTransformer

from embeddings.base_embedding import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using Sentence Transformers

    Args:
        EmbeddingModel (EmbeddingModel): base embedding model
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model

        Args:
            model_name (str, optional): Model to use. Defaults to "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for the given texts

        Args:
            texts (List[str]): List of texts to get embeddings for

        Returns:
            List[List[float]]: List of embeddings
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def __str__(self) -> str:
        """Return string representation of the model

        Returns:
            str: Model name and type
        """
        return f"SentenceTransformerEmbedding({self.model_name})"

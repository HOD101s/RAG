"""ChromaDB implementation for vector database"""

import os
import uuid
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings

from embeddings.base_embedding import EmbeddingModel


class ChromaDb:
    """ChromaDB implementation for vector database"""

    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        persist_directory: Optional[str] = "chroma_db",
    ):
        """Initialize ChromaDB

        Args:
            collection_name (str): Name of the collection
            embedding_model (EmbeddingModel): Embedding model to use
            persist_directory (Optional[str], optional): Directory to persist the
            database. Defaults to "chroma_db".
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Create persist directory if it doesn't exist
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_records(
        self, documents: List[str], ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the vector database.

        Args:
            documents: List of text documents to add to the database
            ids: Optional list of IDs for the documents. If not provided, IDs will be
                generated automatically.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        self.collection.add(documents=documents, ids=ids)

    def query_records(
        self, query: str, n_results: int = 3
    ) -> Dict[str, List[Union[str, float]]]:
        """Query the vector database for similar documents.

        Args:
            query: The query text to search for
            n_results: Number of results to return (default: 3)

        Returns:
            Dictionary containing documents and distances
        """
        return self.collection.query(query_texts=[query], n_results=n_results)

    def reset(self) -> None:
        """Reset the collection"""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

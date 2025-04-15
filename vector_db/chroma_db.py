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
        persist_directory: Optional[str] = "chroma_db"
    ):
        """Initialize ChromaDB

        Args:
            collection_name (str): Name of the collection
            embedding_model (EmbeddingModel): Embedding model to use
            persist_directory (Optional[str], optional): Directory to persist the database. Defaults to "chroma_db".
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_records(self, documents: List[str], ids: Optional[List[str]] = None) -> None:
        """Add records to the collection

        Args:
            documents (List[str]): List of documents to add
            ids (Optional[List[str]], optional): List of IDs for the documents. Defaults to None.
        """
        # Generate embeddings
        embeddings = self.embedding_model.get_embeddings(documents)
        
        # Generate unique IDs if not provided
        if ids is None:
            ids = [f"doc_{uuid.uuid4()}" for _ in range(len(documents))]
        
        # Add documents to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )

    def query_records(
        self, query: str, n_results: int = 5
    ) -> Dict[str, Union[List[List[float]], List[str], List[float]]]:
        """Query the collection

        Args:
            query (str): Query string
            n_results (int, optional): Number of results to return. Defaults to 5.

        Returns:
            Dict[str, Union[List[List[float]], List[str], List[float]]]: Query results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.get_embeddings([query])[0]
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def reset(self) -> None:
        """Reset the collection"""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

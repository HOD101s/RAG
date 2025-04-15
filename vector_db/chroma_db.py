"""Chroma Database as our VectorDb"""

from typing import List, Optional

import chromadb

from embeddings.base_embedding import EmbeddingModel


class ChromaDb:
    """Chroma Database as our VectorDb"""

    def __init__(self, collection_name: str, embedding_model: EmbeddingModel):
        self.collection_name = collection_name
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        self.embedding_model = embedding_model

        # constants
        self.default_id = "id"

    def __get_default_ids(self, expected_size: int) -> List[str]:
        """return list of ids

        Args:
            expected_size (int): number of ids to return

        Returns:
            _type_: List[str]
        """
        return [
            f"{self.default_id}{id_number}" for id_number in range(1, expected_size + 1)
        ]

    def add_records(
        self,
        data: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[str]] = None,
    ):
        """add records to vector db

        Args:
            data (List[str]): list of string documents
            embedding_model (EmbeddingModel): embedding model
            ids (List[str], optional): id for documents. Defaults to None.
            metadatas (List[str], optional): _description_. Defaults to None.
        """
        if ids is None:
            ids = self.__get_default_ids(len(data))

        self.collection.add(
            documents=data,
            embeddings=self.embedding_model.get_embeddings(data),
            ids=ids,
            metadatas=metadatas,
        )

    def query_records(
        self, query: str, query_result_count: int = 5, where: dict = None
    ):
        """query records from vector db

        Args:
            query (str): query string
            query_result_count (int, optional): number of results to return.
            Defaults to 5.
            where (dict, optional): filter results. Defaults to None.

        Returns:
            dict: query results
        """
        return self.collection.query(
            query_embeddings=self.embedding_model.get_embeddings([query]),
            n_results=query_result_count,
            where=where,
        )

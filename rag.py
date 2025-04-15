"""RAG using ChromaDB and Ollama"""

import os
from typing import List

from embeddings.base_embedding import EmbeddingModel
from embeddings.sentence_transformer_embeddings import SentenceTransformerEmbedding
from llm.ollama_llm import OllamaLLM
from vector_db.chroma_db import ChromaDb


def load_data(file_path: str) -> List[str]:
    """Load data from a text file

    Args:
        file_path (str): Path to the text file

    Returns:
        List[str]: List of lines from the file
    """
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_questions(file_path: str) -> List[str]:
    """Load questions from a text file

    Args:
        file_path (str): Path to the text file

    Returns:
        List[str]: List of questions
    """
    return load_data(file_path)


def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Load Star Wars data
    data = load_data("data/star_wars.txt")
    questions = load_questions("data/star_wars_questions.txt")

    # embedding model
    embedding_model: EmbeddingModel = SentenceTransformerEmbedding()

    # chroma db
    vector_db = ChromaDb("star_wars_collection", embedding_model)

    # llm
    llm = OllamaLLM()

    # add records
    vector_db.add_records(data)

    # Ask questions
    for i, question in enumerate(questions[:5], 1):  # Just test with first 5 questions
        print(f"\nQuestion {i}: {question}")
        results = vector_db.query_records(question, 5)["documents"][0]
        response = llm.generate_response(question, results)
        print(f"Answer: {response}")


if __name__ == "__main__":
    main()

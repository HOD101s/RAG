"""FastAPI application for RAG system"""

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Initialize components
    app.state.embedding_model = SentenceTransformerEmbedding()
    app.state.vector_db = ChromaDb("rag_collection", app.state.embedding_model, persist_directory="chroma_db")
    app.state.llm = OllamaLLM()
    
    # Pre-populate the database with Star Wars data if it's empty
    try:
        if app.state.vector_db.collection.count() == 0:
            star_wars_data = load_data("data/star_wars.txt")
            app.state.vector_db.add_records(star_wars_data)
            print(f"Pre-populated database with {len(star_wars_data)} Star Wars facts")
    except Exception as e:
        print(f"Error pre-populating database: {str(e)}")
    
    yield


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation system",
    version="0.1.0",
    lifespan=lifespan
)


class DataInput(BaseModel):
    """Model for data input"""
    documents: List[str]


class QuestionInput(BaseModel):
    """Model for question input"""
    question: str
    num_results: int = 5


class Response(BaseModel):
    """Model for API response"""
    answer: str
    context: List[str]


@app.post("/data", status_code=201)
async def add_data(data_input: DataInput):
    """Add documents to the vector database"""
    try:
        # Add documents to the vector database
        app.state.vector_db.add_records(data_input.documents)
        
        return {"message": f"Added {len(data_input.documents)} documents to the vector database"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/question", response_model=Response)
async def ask_question(question_input: QuestionInput):
    """Ask a question and get a response"""
    try:
        # Query the vector database
        results = app.state.vector_db.query_records(
            question_input.question, 
            question_input.num_results
        )["documents"][0]
        
        # Generate response using the LLM
        response = app.state.llm.generate_response(question_input.question, results)
        
        return Response(
            answer=response,
            context=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
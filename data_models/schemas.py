"""Schemas for the RAG system"""

from typing import List, Optional

from pydantic import BaseModel, Field


class DataInput(BaseModel):
    """Model for inputting data into the RAG system."""

    data: List[str] = Field(
        ..., description="List of text documents to add to the RAG system"
    )


class QuestionInput(BaseModel):
    """Model for asking questions to the RAG system."""

    question: str = Field(..., description="The question to ask the RAG system")
    num_results: int = Field(
        default=3, description="Number of results to return", ge=1, le=20
    )


class Response(BaseModel):
    """Model for the RAG system's output."""

    answer: str = Field(..., description="The generated answer to the question")
    context: Optional[List[str]] = Field(
        None, description="The relevant context used to generate the answer"
    )

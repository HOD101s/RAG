"""LLM using Ollama"""

from typing import List

import requests

from llm.base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """LLM using Ollama"""

    def __init__(
        self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        """Initialize the Ollama LLM

        Args:
            model_name (str, optional): Model to use. Defaults to "llama3.2".
            base_url (str, optional): Ollama API base URL.
                Defaults to "http://localhost:11434".
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using the Ollama API

        Args:
            query (str): User's question
            context (List[str]): Retrieved context documents

        Returns:
            str: Generated response
        """
        # Prepare the prompt with context
        context_str = "\n".join(context)
        prompt = f"""Answer the following question based on the given context.\
            If you cannot answer based on the context, say so.

Context:
{context_str}

Question: {query}

Answer:"""

        # Generate response using Ollama API
        response = requests.post(
            self.api_url,
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            timeout=30,  # 30 second timeout
        )
        response.raise_for_status()
        return response.json()["response"]

    def __str__(self) -> str:
        """Return string representation of the model

        Returns:
            str: Model name and type
        """
        return f"OllamaLLM({self.model_name})"

"""LLM using HuggingFace"""

from typing import List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from llm.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    """LLM using HuggingFace"""

    _instance = None
    _model = None
    _tokenizer = None
    _pipe = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HuggingFaceLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize the HuggingFace LLM

        Args:
            model_name (str, optional): Model to use. Defaults to "google/flan-t5-base".
        """
        if self._model is not None:
            return

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
        )

        self._pipe = pipeline(
            "text2text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_length=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            device_map="auto",
        )

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using the LLM

        Args:
            query (str): User's question
            context (List[str]): Retrieved context documents

        Returns:
            str: Generated response
        """
        # Prepare the prompt with context
        context_str = "\n".join(context)
        prompt = f"""Answer the following question based on the given context. \
             If you cannot answer based on the context, say so. \

Context:
{context_str}

Question: {query}

Answer:"""

        # Generate response
        response = self._pipe(prompt)[0]["generated_text"]
        return response.strip()

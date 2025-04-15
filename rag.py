"""RAG using ChromaDB and Ollama"""

from embeddings.base_embedding import EmbeddingModel
from embeddings.sentence_transformer_embeddings import SentenceTransformerEmbedding
from llm.ollama_llm import OllamaLLM
from vector_db.chroma_db import ChromaDb

# embedding model
embedding_model: EmbeddingModel = SentenceTransformerEmbedding()

# chroma db
vector_db = ChromaDb("first_collection", embedding_model)

# llm
llm = OllamaLLM()

# add records
data = [
    "India is a country in Asia",
    "It has 28 states and 7 union territories",
    "Delhi is the capital of India",
    "Mumbai is the financial capital of India",
    "Kolkata is the cultural capital of India",
    "Chennai is the industrial capital of India",
    "Hyderabad is the technology capital of India",
]

vector_db.add_records(data)

query = "List the various capitals of India?"

results = vector_db.query_records(query, 10)["documents"][0]

response = llm.generate_response(query, results)
print(response)

# RAG (Retrieval-Augmented Generation) System

A simple yet powerful RAG implementation using ChromaDB for vector storage and Ollama for LLM inference.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that:
1. Stores documents in a vector database (ChromaDB)
2. Retrieves relevant context for user queries
3. Generates responses using a local LLM (Ollama)

## Features

- **Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- **Persistent Storage**: Database is saved to disk and persists between restarts
- **Pre-populated Data**: Comes with Star Wars information ready to query
- **Embedding Model**: Uses Sentence Transformers for creating embeddings
- **LLM Integration**: Uses Ollama for local LLM inference (chosen for its quantized models on macOS)
- **Modular Design**: Easy to swap components (e.g., different LLMs, embedding models)
- **REST API**: FastAPI implementation for easy integration
- **Web Interface**: Simple web UI for interacting with the RAG system

## Requirements

- Python 3.8+
- Ollama installed and running locally
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HOD101s/RAG.git
   cd rag
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install and start Ollama:
   ```bash
   # Follow instructions at https://ollama.ai/download
   # Then pull a model:
   ollama pull llama3.2
   ```

## Usage

### Command Line Interface

1. Run the example with Star Wars data:
   ```bash
   python rag.py
   ```

2. To use your own data:
   - Create a text file with your documents (one per line)
   - Create a text file with your questions
   - Update the file paths in `rag.py`

### Web Interface

1. Start the API server:
   ```bash
   python api.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Use the web interface to:
   - Add documents to the vector database
   - Ask questions and get responses
   - View the context used to generate answers

### REST API

1. Start the API server:
   ```bash
   python api.py
   ```

2. Access the API documentation at `http://localhost:8000/docs`

3. API Endpoints:
   - `POST /data`: Add documents to the vector database
   - `POST /question`: Ask a question and get a response

4. Example API usage:
   ```bash
   # Add documents
   curl -X POST "http://localhost:8000/data" \
        -H "Content-Type: application/json" \
        -d '{"documents": ["Star Wars is a space opera franchise.", "Luke Skywalker is the main character."]}'
   
   # Ask a question
   curl -X POST "http://localhost:8000/question" \
        -H "Content-Type: application/json" \
        -d '{"question": "Who is the main character in Star Wars?", "num_results": 5}'
   ```

5. Pre-populated Data:
   - The API automatically loads Star Wars data from `data/star_wars.txt` on startup
   - The data is stored in a persistent database in the `chroma_db` directory
   - You can add more data using the `/data` endpoint

## Project Structure

```
rag/
├── api.py                      # FastAPI implementation
├── chroma_db/                  # Persistent ChromaDB storage
├── data/                       # Data files
│   ├── star_wars.txt           # Star Wars information
│   └── star_wars_questions.txt # Questions about Star Wars
├── embeddings/                 # Embedding models
│   ├── __init__.py
│   ├── base_embedding.py       # Base class for embeddings
│   └── sentence_transformer_embeddings.py  # Sentence Transformer implementation
├── llm/                        # Language models
│   ├── __init__.py
│   ├── base_llm.py             # Base class for LLMs
│   ├── huggingface_llm.py      # HuggingFace implementation
│   └── ollama_llm.py           # Ollama implementation
├── static/                     # Static files for web interface
│   └── index.html              # Web interface
├── vector_db/                  # Vector databases
│   ├── __init__.py
│   └── chroma_db.py            # ChromaDB implementation
├── .flake8                     # Flake8 configuration
├── .gitignore                  # Git ignore file
├── lint.sh                     # Linting script
├── mypy.ini                    # MyPy configuration
├── pyproject.toml              # Black and isort configuration
├── rag.py                      # Main script
└── requirements.txt            # Dependencies
```

## Why Ollama?

We chose Ollama for this project because:
1. It provides quantized models that run efficiently on macOS
2. It's easy to set up and use
3. It supports a wide range of models
4. It runs locally, ensuring privacy and reducing latency

## Extending the Project

### Adding a New Embedding Model

1. Create a new class in the `embeddings` directory
2. Inherit from `EmbeddingModel`
3. Implement the `get_embeddings` method

### Adding a New LLM

1. Create a new class in the `llm` directory
2. Inherit from `BaseLLM`
3. Implement the `generate_response` method

### Adding a New Vector Database

1. Create a new class in the `vector_db` directory
2. Implement methods for adding and querying records

## License

MIT

## Acknowledgements

- [ChromaDB](https://github.com/chroma-core/chroma)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Ollama](https://ollama.ai/)
- [FastAPI](https://fastapi.tiangolo.com/) 
# RAG Document Intelligence API

A production-style Retrieval-Augmented Generation (RAG) backend that ingests documents, retrieves the most relevant passages for a question, and returns an evidence-backed answer with source references.

## Architecture

```
Documents → chunking → TF-IDF vector index → hybrid retrieval → answer synthesis → FastAPI response
```

## Features
- Document ingestion and sentence-aware chunking
- Semantic-style retrieval with TF-IDF cosine similarity
- Keyword overlap bonus for hybrid ranking
- Source citations and confidence score in every answer
- REST API with OpenAPI documentation
- Unit tests and Docker configuration

## Tech stack
Python · FastAPI · scikit-learn · Pydantic · Docker · pytest

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and use:
- `POST /documents` to add a document
- `POST /ask` to query the knowledge base

## Example request

```json
{"question": "How does hybrid retrieval rank document chunks?", "top_k": 3}
```

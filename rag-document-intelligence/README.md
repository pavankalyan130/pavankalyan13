# RAG Document Intelligence API

A production-style Retrieval-Augmented Generation (RAG) backend that ingests documents, retrieves the most relevant passages for a question, and generates an evidence-backed answer with source references.

## Architecture

```
Documents → chunking → TF-IDF vector index → hybrid retrieval → LLM-grounded answer → FastAPI response
```

## Features
- Document ingestion and sentence-aware chunking
- Hybrid retrieval: TF-IDF cosine similarity plus keyword-overlap ranking
- Optional OpenAI LLM synthesis strictly grounded in retrieved sources
- Source citations and confidence score in every answer
- REST API with OpenAPI documentation, tests, and Docker

## Tech stack
Python · FastAPI · OpenAI API · scikit-learn · Pydantic · Docker · pytest

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env
# add your own key to .env or export it in your terminal
uvicorn app.main:app --reload
```

Set `OPENAI_API_KEY` before starting the API. The key is never committed: `.env` is ignored by Git.

Without a key, the API still returns an extractive, source-backed answer. With a key, it uses the configured LLM for a clearer, concise answer based only on retrieved evidence.

Open `http://127.0.0.1:8000/docs` and use:
- `POST /documents` to add a document
- `POST /ask` to query the knowledge base

## Example request

```json
{"question": "How does hybrid retrieval rank document chunks?", "top_k": 3}
```

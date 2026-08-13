import os
from fastapi import FastAPI, HTTPException
from .schemas import Answer, DocumentInput, QuestionInput
from .rag_engine import RAGEngine
from .llm import grounded_answer

app = FastAPI(title="RAG Document Intelligence API", version="1.1.0")
engine = RAGEngine()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": len(engine.chunks),
        "llm_enabled": bool(os.getenv("OPENAI_API_KEY")),
    }

@app.post("/documents")
def ingest_document(document: DocumentInput):
    count = engine.add_document(document.document_id, document.text)
    return {"document_id": document.document_id, "chunks_indexed": count}

@app.post("/ask", response_model=Answer)
def ask_question(question: QuestionInput):
    try:
        retrieval = engine.answer(question.question, question.top_k)
        answer = grounded_answer(question.question, retrieval["sources"], retrieval["answer"])
        return {**retrieval, "answer": answer}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

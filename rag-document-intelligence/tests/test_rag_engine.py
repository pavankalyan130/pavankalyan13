from app.rag_engine import RAGEngine

def test_engine_retrieves_relevant_source():
    engine = RAGEngine()
    engine.add_document(
        "rag-guide",
        "RAG combines retrieval with generation. A vector index retrieves relevant chunks. "
        "Hybrid ranking combines semantic similarity and keyword overlap. "
        "Source citations make answers more trustworthy."
    )
    result = engine.answer("How does hybrid ranking work?")
    assert result["sources"][0]["document_id"] == "rag-guide"
    assert result["confidence"] >= 0

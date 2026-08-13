"""In-memory hybrid retrieval engine for a RAG application."""
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Chunk:
    document_id: str
    chunk_id: int
    text: str

class RAGEngine:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = None

    def _chunk_text(self, text: str, size: int = 3) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences), size)]

    def add_document(self, document_id: str, text: str) -> int:
        new_chunks = [
            Chunk(document_id, index, chunk)
            for index, chunk in enumerate(self._chunk_text(text))
        ]
        self.chunks.extend(new_chunks)
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])
        return len(new_chunks)

    def search(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            raise ValueError("No documents have been ingested.")
        query_vector = self.vectorizer.transform([question])
        semantic_scores = cosine_similarity(query_vector, self.matrix).flatten()
        keywords = set(re.findall(r"\b\w+\b", question.lower()))
        hybrid_scores = []
        for chunk, semantic in zip(self.chunks, semantic_scores):
            overlap = len(keywords & set(re.findall(r"\b\w+\b", chunk.text.lower())))
            hybrid_scores.append(float(semantic) + 0.03 * overlap)
        ranked = sorted(enumerate(hybrid_scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.chunks[index], round(score, 3)) for index, score in ranked]

    def answer(self, question: str, top_k: int = 3) -> dict:
        results = self.search(question, top_k)
        best_chunk, best_score = results[0]
        answer = (
            f"Based on the indexed documents: {best_chunk.text} "
            f"This answer is supported by the highest-ranked source."
        )
        return {
            "answer": answer,
            "confidence": min(round(best_score, 2), 1.0),
            "sources": [
                {"document_id": chunk.document_id, "chunk_id": chunk.chunk_id,
                 "score": score, "excerpt": chunk.text[:220]}
                for chunk, score in results
            ],
        }

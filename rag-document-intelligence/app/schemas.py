from pydantic import BaseModel, Field

class DocumentInput(BaseModel):
    document_id: str = Field(..., min_length=2)
    text: str = Field(..., min_length=20)

class QuestionInput(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=3, ge=1, le=5)

class Source(BaseModel):
    document_id: str
    chunk_id: int
    score: float
    excerpt: str

class Answer(BaseModel):
    answer: str
    confidence: float
    sources: list[Source]

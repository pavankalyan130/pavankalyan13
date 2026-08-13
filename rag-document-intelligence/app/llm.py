"""Optional LLM adapter. It never sends anything except retrieved source text."""
import os

def grounded_answer(question: str, sources: list[dict], fallback: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback

    from openai import OpenAI
    context = "\n\n".join(
        f"[{source['document_id']}#{source['chunk_id']}] {source['excerpt']}"
        for source in sources
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only using the supplied context. If the context is insufficient, "
                    "say so. Keep the answer concise and cite source IDs in square brackets."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
    )
    return response.choices[0].message.content or fallback

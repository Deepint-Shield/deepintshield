"""Evaluate content safety after the application authorizes document access.

For private corpora, enforce user permissions before sending retrieved chunks.
"""

import os

from deepintshield import DeepintShield, build_chunk


shield = DeepintShield.from_env()
litellm = shield.litellm()

query = "Summarize the leave policy."
chunks = [
    build_chunk(
        chunk_id="chunk-1",
        document_id="doc-leave-1",
        content="Employees must submit planned leave at least two weeks in advance.",
        source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"),
        source_name="Leave Policy",
        trust_score=91,
    ),
]

allowed, _ = shield.rag.filter(query=query, chunks=chunks, source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"))
context = "\n\n".join(c.content for c in allowed)

response = litellm.completion(
    model=os.getenv("DEEPINTSHIELD_LITELLM_MODEL", "gpt-4o-mini"),
    messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
)

print("Allowed chunks:", [c.chunk_id for c in allowed])
print(response.choices[0].message.content)

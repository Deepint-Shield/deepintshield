"""Evaluate content safety after the application authorizes document access.

For private corpora, enforce user permissions before sending retrieved chunks.
"""

import os

from deepintshield import DeepintShield, build_chunk


shield = DeepintShield.from_env()
bedrock = shield.bedrock()

query = "Summarize the procurement policy."
chunks = [
    build_chunk(
        chunk_id="chunk-1",
        document_id="doc-procurement-1",
        content="All purchases above $5,000 require a procurement review.",
        source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"),
        source_name="Procurement Policy",
        trust_score=95,
    ),
]

allowed, _ = shield.rag.filter(query=query, chunks=chunks, source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"))
context = "\n\n".join(c.content for c in allowed)

response = bedrock.converse(
    modelId=os.getenv("DEEPINTSHIELD_BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229"),
    messages=[{"role": "user", "content": [{"text": f"Context:\n{context}\n\nQuestion: {query}"}]}],
)

print("Allowed chunks:", [c.chunk_id for c in allowed])
print(response)

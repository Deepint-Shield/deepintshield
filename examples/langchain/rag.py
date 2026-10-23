"""Evaluate content safety after the application authorizes document access.

For private corpora, enforce user permissions before sending retrieved chunks.
"""

import os

from langchain_core.messages import HumanMessage

from deepintshield import DeepintShield, build_chunk


shield = DeepintShield.from_env()
llm = shield.langchain(model="gpt-4o-mini")

query = "Summarize the visitor policy."
chunks = [
    build_chunk(
        chunk_id="chunk-1",
        document_id="doc-visitor-1",
        content="Visitors must sign in and be escorted in restricted areas.",
        source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"),
        source_name="Visitor Policy",
        trust_score=96,
    ),
]

allowed, _ = shield.rag.filter(query=query, chunks=chunks, source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"))
context = "\n\n".join(c.content for c in allowed)

response = llm.invoke([HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")])
print("Allowed chunks:", [c.chunk_id for c in allowed])
print(response.content)

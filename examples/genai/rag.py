import os

from deepintshield import DeepintShield, build_chunk


shield = DeepintShield.from_env()
genai = shield.genai()

query = "Summarize the travel policy."
chunks = [
    build_chunk(
        chunk_id="chunk-1",
        document_id="doc-travel-1",
        content="Flights above economy class require executive approval.",
        source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"),
        source_name="Travel Policy",
        trust_score=90,
    ),
]

allowed, _ = shield.rag.filter(query=query, chunks=chunks, source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"))
context = "\n\n".join(c.content for c in allowed)

response = genai.models.generate_content(
    model=os.getenv("DEEPINTSHIELD_GENAI_MODEL", "gemini-2.5-flash"),
    contents=f"Context:\n{context}\n\nQuestion: {query}",
    config={"automatic_function_calling": {"disable": True}},
)

print("Allowed chunks:", [c.chunk_id for c in allowed])
print(response.text)

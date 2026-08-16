import os

from pydantic_ai import Agent

from deepintshield import DeepintShield, build_chunk


shield = DeepintShield.from_env()
agent = Agent(
    shield.bind("pydanticai").model("gpt-4o-mini"),
    instructions="Answer only from the provided context.",
)

query = "Summarize the access policy."
chunks = [
    build_chunk(
        chunk_id="chunk-1",
        document_id="doc-access-1",
        content="Contractors must use temporary badges and be escorted in secure rooms.",
        source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"),
        source_name="Access Policy",
        trust_score=90,
    ),
]

allowed, _ = shield.rag.filter(query=query, chunks=chunks, source_id=os.getenv("DEEPINTSHIELD_RAG_SOURCE_ID", "kb-public"))
context = "\n\n".join(c.content for c in allowed)

result = agent.run_sync(f"Context:\n{context}\n\nQuestion: {query}")
print("Allowed chunks:", [c.chunk_id for c in allowed])
print(result.output)

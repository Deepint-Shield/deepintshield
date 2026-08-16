"""Pre-embedding input screening: check text for PII / injection / toxicity
*before* it is vectorised. Complements guard_retriever (which filters the
results). A blocking verdict stops before embedding."""
from deepintshield import DeepintShield


shield = DeepintShield.from_env()

# Route embeddings through the gateway, then screen input before embedding.
embedder = shield.bind("langgraph").embedder("text-embedding-3-small")
embedder = shield.rag.guard_embedder(embedder)

vec = embedder.embed_query("My SSN is 123-45-6789")
print("embedded, dims:", len(vec))

"""LlamaIndex with both the LLM and the embedding model routed through
DeepintShield via the global Settings."""
from llama_index.core import Settings

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
Settings.llm = shield.bind("llamaindex").llm("gpt-4o-mini")
Settings.embed_model = shield.bind("llamaindex").embedder("text-embedding-3-small")

print(Settings.llm.complete("Say hello from LlamaIndex via DeepintShield."))

"""Native LangChain tool call with automatic Agentic enforcement.

Constructing the client arms LangChain's ordinary tool boundary. The tool stays
third-party code: no DeepintShield callback, wrapper, or exception formatter is
required.
"""

from langchain_core.tools import tool

from deepintshield import DeepintShield


shield = DeepintShield.from_env()


@tool
def write_ledger(row: str) -> str:
    """Append a row to the finance ledger."""
    return f"wrote {row}"


print(write_ledger.invoke({"row": "amount=12.5"}))

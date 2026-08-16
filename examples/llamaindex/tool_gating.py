"""Native LlamaIndex FunctionTool call with automatic Agentic enforcement."""
from llama_index.core.tools import FunctionTool

from deepintshield import DeepintShield


shield = DeepintShield.from_env()


def transfer_funds(amount: float) -> str:
    """Transfer funds between accounts."""
    return f"transferred {amount}"


tool = FunctionTool.from_defaults(fn=transfer_funds)
print(tool.call(amount=10.0))

"""Gate every tool node in a compiled LangGraph through the PDP. One line,
no graph-shape change - existing invoke() code is unchanged."""
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
llm = shield.langgraph().model("gpt-4o-mini")


@tool
def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    return f"deleted {path}"


graph = create_react_agent(llm, [delete_file])
graph = shield.agentic.langgraph(graph)  # every tool node now PEP-gated

print(graph.invoke({"messages": [("user", "delete /tmp/report.csv")]}))

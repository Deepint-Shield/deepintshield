"""Ordinary custom LangGraph nodes with automatic Agentic enforcement."""
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from deepintshield import DeepintShield


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state: AgentState):
    return {"messages": [AIMessage(content=f"Responding to: {state['messages'][-1].content}")]}


shield = DeepintShield.from_env()

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Summarize the policy safely.")]})
print(result["messages"][-1].content)

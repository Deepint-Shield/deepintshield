"""Native LangGraph workflow with automatic Agentic enforcement."""
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from deepintshield import DeepintShield


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state: AgentState):
    return {
        "messages": [
            AIMessage(
                content="Searching the policy handbook.",
                tool_calls=[{
                    "name": "knowledge_search",
                    "args": {"query": state["messages"][-1].content},
                    "action_class": "read",
                }],
            )
        ]
    }


def tools_node(_state: AgentState):
    return {"messages": [AIMessage(content="Visitors must remain escorted in secure areas.")]}


shield = DeepintShield.from_env()

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", END)

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Find the visitor policy.")]})
print(result["messages"][-1].content)

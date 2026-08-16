"""Native LangGraph discovery and execution.

Set ``DEEPINTSHIELD_AGENT_NAME`` and ``DEEPINTSHIELD_REQUESTER`` in the
environment. A successful first execution capture for an unseen agent emits the
stable code ``agent_registration_pending``; a failed capture remains
``agent_not_registered``. Complete registration in the Agentic interface, then
run the same application again.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from deepintshield import DeepintShield


class State(TypedDict):
    query: str
    result: str


def location_extract(state: State) -> State:
    return {"query": state["query"], "result": "acme-hq"}


def crm_read(state: State) -> State:
    return {"query": state["query"], "result": f"account for {state['result']}"}


shield = DeepintShield.from_env()

builder = StateGraph(State)
builder.add_node("location_extract", location_extract)
builder.add_node("crm_read", crm_read)
builder.add_edge(START, "location_extract")
builder.add_edge("location_extract", "crm_read")
builder.add_edge("crm_read", END)
app = builder.compile()

print(app.invoke({"query": "acme corp", "result": ""})["result"])

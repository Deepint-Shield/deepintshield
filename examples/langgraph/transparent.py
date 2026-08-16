"""LangGraph / LangChain native model via the DeepintShield binder (drop-in).
Use the returned ChatOpenAI anywhere you'd use a normal one."""
from langchain_core.messages import HumanMessage

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
llm = shield.bind("langgraph").model("gpt-4o-mini")

print(llm.invoke([HumanMessage(content="hello from the LangGraph binder")]).content)

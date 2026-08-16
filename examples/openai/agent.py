"""Explicit input, tool, and output guardrail stages."""
from deepintshield import DeepintShield


shield = DeepintShield.from_env()


@shield.agent.tool(action_class="read")
def knowledge_search(query: str) -> str:
    return f"Policy excerpt about: {query}"


user_input = "Find the visitor policy."

shield.agent.check_input(user_input)
tool_result = knowledge_search(user_input)
shield.agent.check_output(tool_result)
print(tool_result)

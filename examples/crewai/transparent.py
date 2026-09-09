"""CrewAI with all model traffic routed through DeepintShield. Native CrewAI
code - the only DeepintShield line is the LLM binder."""
import os

from crewai import Agent, Crew, Task

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
llm = shield.bind("crewai").llm(os.getenv("DEEPINTSHIELD_MODEL", "openai/gpt-4o-mini"))

researcher = Agent(role="Researcher", goal="Answer concisely", backstory="Expert.", llm=llm)
task = Task(
    description="Summarise Hong Kong in one sentence.",
    expected_output="One sentence.",
    agent=researcher,
)
print(Crew(agents=[researcher], tasks=[task]).kickoff())

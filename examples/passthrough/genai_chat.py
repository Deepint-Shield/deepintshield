import os

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
genai = shield.genai(passthrough=True)

response = genai.models.generate_content(
    model=os.getenv("DEEPINTSHIELD_GENAI_MODEL", "gemini-2.5-flash"),
    contents="Hello from GenAI passthrough.",
    config={"automatic_function_calling": {"disable": True}},
)

print(response.text)

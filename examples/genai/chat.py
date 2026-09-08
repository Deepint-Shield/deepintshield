import os

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
genai = shield.genai()

response = genai.models.generate_content(
    model=os.getenv("DEEPINTSHIELD_GENAI_MODEL", "gemini-2.5-flash"),
    contents="Say hello from Google GenAI via DeepintShield.",
)

print(response.text)

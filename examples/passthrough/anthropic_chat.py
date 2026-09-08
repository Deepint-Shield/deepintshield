import os

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
anthropic = shield.anthropic(passthrough=True)

response = anthropic.messages.create(
    model=os.getenv("DEEPINTSHIELD_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello from Anthropic passthrough."}],
)

print(response.content[0].text)

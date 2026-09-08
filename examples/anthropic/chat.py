import os

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
anthropic = shield.anthropic()

response = anthropic.messages.create(
    model=os.getenv("DEEPINTSHIELD_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    max_tokens=256,
    messages=[{"role": "user", "content": "Say hello from Anthropic via DeepintShield."}],
)

print(response.content[0].text)

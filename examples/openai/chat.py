"""Chat Completions example. For GPT-6 Astra, run responses.py instead:
gateway-injected MCP tools also require Astra's Responses API.
"""

import os

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
openai = shield.openai()

response = openai.chat.completions.create(
    model=os.getenv("DEEPINTSHIELD_MODEL", "gpt-4o-mini"),
    messages=[{"role": "user", "content": "Give me a one sentence summary of Hong Kong."}],
)

print(response.choices[0].message.content)

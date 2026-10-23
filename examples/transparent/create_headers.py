"""Gateway header injector. Drop `create_headers()` into any SDK's
default_headers and point base_url at the gateway route for that provider."""
from openai import OpenAI

from deepintshield import DeepintShield


shield = DeepintShield.from_env()

client = OpenAI(
    base_url=shield.endpoint("openai"),
    api_key=shield.api_key(),
    default_headers=shield.create_headers(),  # provider="openai" by default
)
print(
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi via create_headers()"}],
    ).choices[0].message.content
)

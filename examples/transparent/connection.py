"""The framework-agnostic minimal wrapper: point ANY OpenAI-compatible client
at the gateway with two values - base_url + headers. Everything else (your
prompts, tools, app code) stays 100% native."""
from openai import OpenAI

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
base_url, headers = shield.connection()  # ("…/openai", {x-bf-vk, x-bf-app, …})

client = OpenAI(base_url=base_url, api_key=shield.api_key(), default_headers=headers)
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello via shield.connection()"}],
)
print(resp.choices[0].message.content)

"""Universal pattern for non-Python platforms (n8n, Flowise, Dify, raw HTTP):
set the model node's Base URL + API key to the gateway. Shown here with a raw
httpx call - exactly what those platforms do under the hood. No SDK needed,
guardrails + observability run server-side."""
import httpx

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
base_url, headers = shield.connection()

resp = httpx.post(
    f"{base_url}/chat/completions",
    headers=headers,
    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
)
print(resp.json()["choices"][0]["message"]["content"])

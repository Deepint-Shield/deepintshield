"""GPT-6 Astra through the gateway's native OpenAI Responses API.

Install: pip install 'deepintshield[openai]'
Set DEEPINTSHIELD_VIRTUAL_KEY and optionally DEEPINTSHIELD_BASE_URL.
Run: python examples/openai/responses.py [--stream]

Astra tools, including gateway-injected MCP tools, require Responses.
Use low/medium/high/xhigh/max reasoning; omit temperature and top_p.
Official guidance: https://developers.openai.com/api/docs/guides/latest-model
"""

import argparse
import os

from deepintshield import DeepintShield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true", help="Print text as Responses events arrive.")
    args = parser.parse_args()

    # Credentials stay in environment configuration; no provider secret is
    # needed in this script because the gateway selects its configured key.
    with DeepintShield.from_env() as shield, shield.openai() as client:
        response = client.responses.create(
            model=os.getenv("DEEPINTSHIELD_MODEL", "gpt-6-astra"),
            input="Give me a one sentence summary of Hong Kong.",
            reasoning={"effort": "low"},
            store=False,
            stream=args.stream,
        )
        if args.stream:
            with response as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                    elif event.type in {"response.failed", "response.incomplete", "error"}:
                        raise RuntimeError(f"Responses did not complete: {event.type}")
            print()
        else:
            if response.status != "completed":
                raise RuntimeError(f"Responses did not complete: {response.status}")
            print(response.output_text)


if __name__ == "__main__":
    main()

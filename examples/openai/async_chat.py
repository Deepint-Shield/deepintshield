"""Native AsyncOpenAI; connection uses the existing VK/base URL environment."""

import asyncio
import os

from deepintshield import DeepintShield


async def main() -> None:
    with DeepintShield.from_env() as shield:
        async with shield.async_openai() as client:
            response = await client.chat.completions.create(
                model=os.getenv("DEEPINTSHIELD_MODEL", "openai/gpt-4o-mini"),
                messages=[{"role": "user", "content": "Hello"}],
            )
            print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())

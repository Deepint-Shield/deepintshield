"""Native Temporal orchestration; OpenAI I/O runs only inside an activity.

Install deepintshield[temporal] and start a Temporal service on localhost:7233.
Configure the existing VK/base URL environment plus the governed agent name.
Start InferenceWorkflow on queue inference-tq using your usual Temporal client.
"""

import asyncio
import os
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from deepintshield import DeepintShield


@activity.defn
async def infer(prompt: str) -> str:
    with DeepintShield.from_env() as shield:
        async with shield.async_openai() as client:
            result = await client.chat.completions.create(
                model=os.getenv("DEEPINTSHIELD_MODEL", "openai/gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
            )
            return result.choices[0].message.content or ""


@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return await workflow.execute_activity(infer, prompt, start_to_close_timeout=timedelta(minutes=2))


async def main() -> None:
    # Construct outside the deterministic workflow to install the existing
    # activity governance interceptor before the native worker is created.
    with DeepintShield.from_env():
        client = await Client.connect("localhost:7233")
        async with Worker(client, task_queue="inference-tq", workflows=[InferenceWorkflow], activities=[infer]):
            await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())

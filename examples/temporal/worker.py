"""Temporal durable agent governed automatically at its activity boundary.

Constructing ``DeepintShield`` patches ``Worker`` to inject its activity
interceptor. Every activity is gated outside the deterministic workflow sandbox
(network I/O is allowed there); a DENY becomes a non-retryable
``ApplicationError`` so Temporal does not spin forever on a policy denial.

The only Agentic enforcement line is ``DeepintShield.from_env()``. Everything
else - durability, retries, replay and event-history audit - stays native
Temporal. Point the model activity's OpenAI client at ``shield.openai()`` to add
cache/guardrails/cost on the LLM leg.

    pip install 'deepintshield[temporal]'   # temporalio
"""
import asyncio

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

from deepintshield import DeepintShield

shield = DeepintShield.from_env()


@activity.defn
async def crm_write(record: str) -> str:
    """A tool activity - governed by its function name ``crm_write``."""
    return f"wrote {record}"


@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, record: str) -> str:
        return await workflow.execute_activity(
            crm_write, record, start_to_close_timeout=__import__("datetime").timedelta(seconds=30)
        )


async def main() -> None:
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="agent-tq",
        workflows=[AgentWorkflow],
        activities=[crm_write],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

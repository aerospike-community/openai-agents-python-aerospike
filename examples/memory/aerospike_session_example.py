"""
Example demonstrating Aerospike session memory with a shared client.

In production you should create one aerospike.Client and pass it to all sessions
so they share the same connection pool.
"""

import asyncio

import aerospike

from agents import Agent, Runner
from agents.extensions.memory import AerospikeSession

CONFIG = {"hosts": [("127.0.0.1", 3000)]}
NAMESPACE = "test"


async def main():
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
    )

    # One client shared across all sessions (production pattern).
    try:
        client = aerospike.client(CONFIG).connect()
    except Exception:
        print("Aerospike is not available on localhost:3000")
        print(
            "Start it with: docker run -d --ulimit nofile=20000:20000 "
            '-p 3000:3000 -e "NAMESPACE=test" aerospike/aerospike-server:latest'
        )
        return

    session_a = AerospikeSession("conversation_a", client=client, namespace=NAMESPACE)
    session_b = AerospikeSession("conversation_b", client=client, namespace=NAMESPACE)

    # Clean slate for the demo.
    await session_a.clear_session()
    await session_b.clear_session()

    # --- Session A: multi-turn conversation ---
    print("=== Session A ===")
    result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session_a)
    print(f"Turn 1: {result.final_output}")

    result = await Runner.run(agent, "What state is it in?", session=session_a)
    print(f"Turn 2: {result.final_output}")

    result = await Runner.run(agent, "What's the population of that state?", session=session_a)
    print(f"Turn 3: {result.final_output}")

    # --- Session B: independent conversation on the same client ---
    print("\n=== Session B ===")
    result = await Runner.run(agent, "What is the capital of France?", session=session_b)
    print(f"Turn 1: {result.final_output}")

    # Show isolation.
    a_items = await session_a.get_items()
    b_items = await session_b.get_items()
    print(f"\nSession A items: {len(a_items)}, Session B items: {len(b_items)}")

    # Cleanup.
    await session_a.clear_session()
    await session_b.clear_session()
    await asyncio.to_thread(client.close)


if __name__ == "__main__":
    # pip install "openai-agents[aerospike]"
    asyncio.run(main())

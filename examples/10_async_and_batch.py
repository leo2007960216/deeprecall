"""DeepRecall Async Usage -- Non-blocking queries with AsyncDeepRecall.

Demonstrates:
  - AsyncDeepRecall (non-blocking queries in an event loop)
  - Async document ingestion
  - Async context manager for cleanup
  - Sequential multi-query pattern

Prerequisites:
    pip install deeprecall[chroma]
    export OPENAI_API_KEY=sk-...
"""

import asyncio
import os
import time

from dotenv import load_dotenv

from deeprecall import AsyncDeepRecall
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()


def setup_store() -> ChromaStore:
    """Create and populate a vector store."""
    store = ChromaStore(collection_name="async_demo_v2")
    store.add_documents(
        documents=[
            "The human brain contains approximately 86 billion neurons, each forming "
            "thousands of synaptic connections.",
            "Deep learning neural networks are loosely inspired by the brain's structure, "
            "using layers of artificial neurons to learn patterns.",
            "The hippocampus plays a crucial role in memory formation and spatial navigation.",
            "Transformers replaced RNNs by using self-attention mechanisms, enabling "
            "parallel processing and better long-range dependency handling.",
            "Neuroplasticity allows the brain to reorganize itself by forming new neural "
            "connections throughout life.",
        ],
        metadatas=[
            {"domain": "neuroscience"},
            {"domain": "ai"},
            {"domain": "neuroscience"},
            {"domain": "ai"},
            {"domain": "neuroscience"},
        ],
    )
    return store


async def main() -> None:
    store = setup_store()

    # --- Example 1: Single async query ----------------------------------------
    print("=" * 70)
    print("EXAMPLE 1: Single Async Query")
    print("=" * 70)

    async with AsyncDeepRecall(
        vectorstore=store,
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    ) as engine:
        start = time.perf_counter()
        result = await engine.query("What is neuroplasticity?")
        elapsed = time.perf_counter() - start

        print(f"\n  Answer: {result.answer[:200]}...")
        print(f"  Time: {elapsed:.1f}s | Sources: {len(result.sources)}")

    # --- Example 2: Multiple sequential async queries -------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multiple Async Queries (sequential)")
    print("=" * 70)

    questions = [
        "How many neurons does the human brain have?",
        "How do Transformers improve on RNNs?",
    ]

    async with AsyncDeepRecall(
        vectorstore=store,
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    ) as engine:
        total_start = time.perf_counter()
        for q in questions:
            start = time.perf_counter()
            result = await engine.query(q)
            elapsed = time.perf_counter() - start
            print(f"\n  Q: {q}")
            print(f"  A: {result.answer[:150]}...")
            print(f"  Time: {elapsed:.1f}s")
        total_elapsed = time.perf_counter() - total_start
        print(f"\n  Total time for {len(questions)} queries: {total_elapsed:.1f}s")

    # --- Example 3: Async document ingestion ----------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Async Document Ingestion + Query")
    print("=" * 70)

    async with AsyncDeepRecall(
        vectorstore=store,
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    ) as engine:
        ids = await engine.add_documents(
            [
                "Spiking Neural Networks (SNNs) mimic biological neurons more closely "
                "by using discrete spikes for information transmission."
            ],
            metadatas=[{"domain": "ai"}],
        )
        if ids:
            print(f"  Added document with ID: {ids[0]}")
        else:
            print("  Warning: add_documents returned no IDs")

        result = await engine.query("What are Spiking Neural Networks?")
        print(f"  Answer: {result.answer[:200]}...")

    print("\nAll async examples completed!")


if __name__ == "__main__":
    asyncio.run(main())

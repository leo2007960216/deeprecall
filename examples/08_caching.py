"""DeepRecall Caching -- Avoid redundant LLM calls.

Demonstrates:
  - InMemoryCache (fastest, ephemeral -- good for dev/testing)
  - DiskCache (SQLite-backed, persists across restarts)
  - RedisCache (distributed, production-ready)

Prerequisites:
    pip install deeprecall[chroma]
    export OPENAI_API_KEY=sk-...

For Redis example:
    pip install deeprecall[chroma,redis]
    docker run -d -p 6379:6379 redis:7
"""

import os
import time

from dotenv import load_dotenv

from deeprecall import DeepRecall, DeepRecallConfig, DiskCache, InMemoryCache, RedisCache
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()

# --- Shared vector store setup ------------------------------------------------
store = ChromaStore(collection_name="cache_demo")
store.add_documents(
    documents=[
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "Sound travels at about 343 meters per second in dry air at 20 degrees Celsius.",
        "The Voyager 1 spacecraft, launched in 1977, is the most distant human-made object.",
        "Light from the Sun takes approximately 8 minutes and 20 seconds to reach Earth.",
    ],
    metadatas=[
        {"topic": "physics", "subtopic": "light"},
        {"topic": "physics", "subtopic": "sound"},
        {"topic": "space", "subtopic": "exploration"},
        {"topic": "space", "subtopic": "solar_system"},
    ],
)


def run_cached_query(engine: DeepRecall, label: str) -> None:
    """Run the same query twice and show the speed difference."""
    question = "How fast does light travel compared to sound?"

    print(f"\n--- {label} ---")
    start = time.perf_counter()
    result1 = engine.query(question)
    t1 = time.perf_counter() - start
    print(f"  First query:  {t1:.2f}s | Answer: {result1.answer[:80]}...")

    start = time.perf_counter()
    result2 = engine.query(question)
    t2 = time.perf_counter() - start
    print(f"  Second query: {t2:.2f}s (cache hit!) | Answer: {result2.answer[:80]}...")
    safe_t2 = max(t2, 1e-9)
    print(f"  Speedup: {t1 / safe_t2:.1f}x faster from cache")


# --- Example 1: InMemoryCache ------------------------------------------------
print("=" * 70)
print("EXAMPLE 1: In-Memory Cache")
print("=" * 70)

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    cache=InMemoryCache(max_size=100, default_ttl=3600),
)
with DeepRecall(vectorstore=store, config=config) as engine:
    run_cached_query(engine, "InMemoryCache")

# --- Example 2: DiskCache (SQLite) -------------------------------------------
print("\n" + "=" * 70)
print("EXAMPLE 2: Disk Cache (SQLite)")
print("=" * 70)

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    cache=DiskCache(db_path="./deeprecall_cache_demo.db"),
)
with DeepRecall(vectorstore=store, config=config) as engine:
    run_cached_query(engine, "DiskCache (SQLite)")

# --- Example 3: RedisCache ----------------------------------------------------
# Requires: pip install deeprecall[redis] and a running Redis instance.
# Comment out the block below if Redis is not available on localhost:6379.

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    cache=RedisCache(url="redis://localhost:6379/0"),
)
with DeepRecall(vectorstore=store, config=config) as engine:
    run_cached_query(engine, "RedisCache")

print("\nDone! The DiskCache file is at ./deeprecall_cache_demo.db")

"""DeepRecall Callbacks & Monitoring -- Hook into the reasoning loop.

Demonstrates:
  1. ConsoleCallback + JSONLCallback (live output + structured log files)
  2. UsageTrackingCallback (aggregate usage stats across queries)
  3. Custom callback (subclass BaseCallback to track searches)
  4. ProgressCallback (capture the full event stream for inspection)
  5. OpenTelemetryCallback (distributed tracing -- commented out)

Prerequisites:
    pip install deeprecall[chroma,rich]
    export OPENAI_API_KEY=sk-...

For OpenTelemetry:
    pip install deeprecall[chroma,otel]
"""

import os

from dotenv import load_dotenv

from deeprecall import (
    ConsoleCallback,
    DeepRecall,
    DeepRecallConfig,
    JSONLCallback,
)
from deeprecall.core.callbacks import BaseCallback, ProgressCallback, UsageTrackingCallback
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()

# --- Shared setup -------------------------------------------------------------
store = ChromaStore(collection_name="callbacks_demo")
store.add_documents(
    documents=[
        "Kubernetes (K8s) is an open-source container orchestration platform "
        "originally designed by Google and now maintained by the CNCF.",
        "Docker introduced containerization to mainstream development in 2013, "
        "enabling consistent environments from dev to production.",
        "Terraform by HashiCorp is an infrastructure-as-code tool that supports "
        "multiple cloud providers (AWS, GCP, Azure) with a declarative syntax.",
        "Prometheus is an open-source monitoring system with a dimensional data "
        "model, flexible query language (PromQL), and alerting capabilities.",
        "GitOps is a practice where Git is the single source of truth for "
        "declarative infrastructure and application configuration.",
    ],
    metadatas=[
        {"tool": "kubernetes", "category": "orchestration"},
        {"tool": "docker", "category": "containerization"},
        {"tool": "terraform", "category": "iac"},
        {"tool": "prometheus", "category": "monitoring"},
        {"tool": "gitops", "category": "practice"},
    ],
)


# --- Example 1: Console + JSONL callbacks -------------------------------------
print("=" * 70)
print("EXAMPLE 1: ConsoleCallback + JSONLCallback")
print("=" * 70)

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    callbacks=[
        ConsoleCallback(),
        JSONLCallback(log_dir="./logs"),
    ],
)
with DeepRecall(vectorstore=store, config=config) as engine:
    result = engine.query("What is the modern DevOps stack and how do the tools fit together?")
    print(f"\nAnswer: {result.answer[:200]}...")
    print("\nJSONL logs written to ./logs/")


# --- Example 2: UsageTrackingCallback ----------------------------------------
print("\n" + "=" * 70)
print("EXAMPLE 2: UsageTrackingCallback -- aggregate stats across queries")
print("=" * 70)

tracker = UsageTrackingCallback()
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    callbacks=[tracker],
)
with DeepRecall(vectorstore=store, config=config) as engine:
    engine.query("What is Kubernetes?")
    engine.query("How does Terraform work?")
    engine.query("Explain GitOps.")

    print(f"  Total queries: {tracker.total_queries}")
    print(f"  Total tokens:  {tracker.total_tokens}")
    print(f"  Total time:    {tracker.total_time:.2f}s")


# --- Example 3: Custom callback -----------------------------------------------
print("\n" + "=" * 70)
print("EXAMPLE 3: Custom callback -- track search queries")
print("=" * 70)


class SearchLoggerCallback(BaseCallback):
    """Logs every search the LLM makes."""

    def __init__(self) -> None:
        self.searches: list[dict] = []

    def on_search(self, query: str, num_results: int, time_ms: float) -> None:
        self.searches.append({"query": query, "num_results": num_results})
        print(f"  [SEARCH] '{query}' -> {num_results} results ({time_ms:.0f}ms)")


search_logger = SearchLoggerCallback()
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    callbacks=[search_logger],
)
with DeepRecall(vectorstore=store, config=config) as engine:
    engine.query("Compare Kubernetes and Docker for container management.")

    print(f"\n  Total searches made: {len(search_logger.searches)}")
    for s in search_logger.searches:
        print(f"    - '{s['query']}' ({s['num_results']} results)")


# --- Example 4: ProgressCallback -- accumulate all events --------------------
print("\n" + "=" * 70)
print("EXAMPLE 4: ProgressCallback -- capture full event stream")
print("=" * 70)

progress = ProgressCallback()
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    callbacks=[progress],
)
with DeepRecall(vectorstore=store, config=config) as engine:
    engine.query("What is Prometheus used for?")

    print(f"  Captured {len(progress.events)} events:")
    for ev in progress.events:
        print(f"    - {ev['type']}")


# --- Example 5: OpenTelemetry (uncomment to use) -----------------------------
# Requires: pip install deeprecall[otel]
# Start a local Jaeger: docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one

# from deeprecall import OpenTelemetryCallback
#
# otel = OpenTelemetryCallback(
#     service_name="deeprecall-demo",
#     endpoint="http://localhost:4317",
# )
# config = DeepRecallConfig(
#     backend="openai",
#     backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
#     callbacks=[otel],
# )
# with DeepRecall(vectorstore=store, config=config) as engine:
#     engine.query("What is Prometheus monitoring?")
# # View traces at http://localhost:16686

"""DeepRecall Production-Ready Setup -- Everything you need for production.

Combines:
  - Exception handling (structured error recovery)
  - Retry with exponential backoff
  - Caching (DiskCache for persistence)
  - Callbacks (JSONL logging + usage tracking)
  - Budget guardrails (cost/time limits)
  - Context managers for cleanup

Prerequisites:
    pip install deeprecall[chroma,rich]
    export OPENAI_API_KEY=sk-...
"""

import os
import sys

from dotenv import load_dotenv

from deeprecall import (
    DeepRecall,
    DeepRecallConfig,
    DeepRecallError,
    DiskCache,
    JSONLCallback,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    QueryBudget,
    RetryConfig,
    VectorStoreError,
)
from deeprecall.core.callbacks import UsageTrackingCallback
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()

# =============================================================================
# 1. Setup: vector store with real-world data
# =============================================================================
print("Setting up production-ready DeepRecall...")

store = ChromaStore(collection_name="production_demo", persist_directory="./prod_chroma_db")
store.add_documents(
    documents=[
        "Our authentication system uses JWT tokens with RS256 signing. Access tokens "
        "expire after 15 minutes, refresh tokens after 7 days.",
        "The API gateway handles rate limiting at 100 req/s per user using a token "
        "bucket algorithm. Burst capacity is 150 requests.",
        "Database connections are pooled with a max of 20 connections per service "
        "instance. Connection timeout is 5 seconds.",
        "All PII data is encrypted at rest using AES-256-GCM and in transit using "
        "TLS 1.3. Encryption keys are stored in AWS KMS.",
        "The caching layer uses Redis with a 5-minute TTL for API responses and a "
        "1-hour TTL for user session data.",
        "Deployment uses blue-green strategy with automated canary analysis. Rollback "
        "is triggered if error rate exceeds 1% for 5 minutes.",
        "Logging follows structured JSON format with correlation IDs. All logs ship "
        "to Elasticsearch via Fluentd within 30 seconds.",
        "The CI/CD pipeline runs on GitHub Actions with stages: lint, unit test, "
        "integration test, security scan, deploy to staging, deploy to prod.",
    ],
    metadatas=[
        {"system": "auth", "criticality": "high"},
        {"system": "api_gateway", "criticality": "high"},
        {"system": "database", "criticality": "high"},
        {"system": "security", "criticality": "critical"},
        {"system": "caching", "criticality": "medium"},
        {"system": "deployment", "criticality": "high"},
        {"system": "observability", "criticality": "medium"},
        {"system": "cicd", "criticality": "medium"},
    ],
)

# =============================================================================
# 2. Configure: all production features enabled
# =============================================================================
tracker = UsageTrackingCallback()

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    max_iterations=10,
    top_k=5,
    verbose=False,
    # Retry transient failures automatically
    retry=RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        jitter=True,
    ),
    # Persistent cache -- survives restarts
    cache=DiskCache(db_path="./prod_cache.db"),
    # Callbacks -- structured logging + usage tracking
    callbacks=[
        JSONLCallback(log_dir="./prod_logs"),
        tracker,
    ],
    # Default budget for all queries (can be overridden per-query)
    budget=QueryBudget(
        max_search_calls=15,
        max_tokens=100000,
        max_time_seconds=60.0,
    ),
)

# =============================================================================
# 3. Query: with structured exception handling
# =============================================================================
questions = [
    "How does the authentication system work and what are the token lifetimes?",
    "Describe the full deployment pipeline from code commit to production.",
    "What security measures protect user data at rest and in transit?",
]

with DeepRecall(vectorstore=store, config=config) as engine:
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}/{len(questions)}: {question}")
        print("=" * 60)

        try:
            result = engine.query(
                question,
                budget=QueryBudget(
                    max_search_calls=8,
                    max_time_seconds=30.0,
                ),
            )

            print(f"Answer: {result.answer[:200]}...")
            print(f"Sources: {len(result.sources)} | Time: {result.execution_time:.1f}s")
            print(f"Confidence: {result.confidence}")
            if result.budget_status:
                print(f"Budget: {result.budget_status}")

        except LLMRateLimitError as e:
            print(f"Rate limited! Retry after {e.retry_after}s")
        except LLMTimeoutError:
            print("LLM call timed out -- consider increasing max_time_seconds")
        except LLMProviderError as e:
            print(f"LLM error: {e}")
        except VectorStoreError as e:
            print(f"Vector DB error: {e}")
        except DeepRecallError as e:
            print(f"DeepRecall error: {e}")

# =============================================================================
# 4. Report: aggregate usage stats
# =============================================================================
print("\n" + "=" * 60)
print("USAGE REPORT")
print("=" * 60)
print(f"Total queries:  {tracker.total_queries}")
print(f"Total tokens:   {tracker.total_tokens:,}")
print(f"Total time:     {tracker.total_time:.2f}s")
print(f"Avg per query:  {tracker.total_time / max(tracker.total_queries, 1):.2f}s")
print("\nLogs written to: ./prod_logs/")
print("Cache stored at: ./prod_cache.db")
print("Vector DB at:    ./prod_chroma_db/")

sys.exit(0)

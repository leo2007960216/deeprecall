"""DeepRecall with Qdrant -- High-performance vector similarity search.

Demonstrates:
  - QdrantStore with a custom embedding function
  - Distance metric selection (Cosine, Euclid, Dot)
  - Document CRUD (add, search, delete, count)
  - Filtered search with metadata

Prerequisites:
    pip install deeprecall[qdrant]
    # Start Qdrant: docker run -d -p 6333:6333 qdrant/qdrant
    export OPENAI_API_KEY=sk-...
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from deeprecall import DeepRecall
from deeprecall.vectorstores.qdrant import QdrantStore

load_dotenv()

# --- Embedding function -------------------------------------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_fn(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


# --- Setup Qdrant store ------------------------------------------------------
store = QdrantStore(
    collection_name="qdrant_demo",
    url="http://localhost:6333",
    dimension=1536,
    distance="Cosine",
    embedding_fn=embed_fn,
)

# --- Add documents ------------------------------------------------------------
print("Adding documents...")
ids = store.add_documents(
    documents=[
        "PostgreSQL is an open-source relational database known for its reliability "
        "and feature richness, supporting JSON, full-text search, and extensions.",
        "MongoDB is a document-oriented NoSQL database that stores data in flexible "
        "BSON format, ideal for rapidly changing schemas.",
        "Redis is an in-memory key-value store used for caching, session management, "
        "and real-time leaderboards, with sub-millisecond latency.",
        "Apache Cassandra is a distributed NoSQL database designed for handling large "
        "amounts of data across commodity servers with no single point of failure.",
        "SQLite is a serverless, file-based SQL database engine that is embedded "
        "directly into applications, commonly used in mobile and desktop apps.",
    ],
    metadatas=[
        {"type": "relational", "use_case": "general"},
        {"type": "document", "use_case": "flexible_schema"},
        {"type": "key_value", "use_case": "caching"},
        {"type": "wide_column", "use_case": "distributed"},
        {"type": "embedded", "use_case": "local"},
    ],
)
print(f"Added {len(ids)} documents: {ids}")
print(f"Total count: {store.count()}")

# --- Search -------------------------------------------------------------------
print("\n--- Vector search ---")
results = store.search("What database is best for caching?", top_k=3)
for r in results:
    print(f"  Score: {r.score:.3f} | {r.content[:80]}...")

# --- Query with DeepRecall engine ---------------------------------------------
print("\n--- DeepRecall recursive reasoning ---")
with DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
) as engine:
    result = engine.query("Compare relational and NoSQL databases. When should you use each type?")
    print(f"Answer: {result.answer[:300]}...")
    print(f"Sources: {len(result.sources)}, Time: {result.execution_time:.1f}s")

# --- Delete documents ---------------------------------------------------------
print("\n--- Deleting first 2 documents ---")
store.delete(ids[:2])
print(f"Count after delete: {store.count()}")

# --- Cleanup ------------------------------------------------------------------
store.close()
print("\nDone! Qdrant connection closed.")

"""DeepRecall with FAISS -- Fast local vector search for ML workflows.

Demonstrates:
  - FAISSStore setup with a custom embedding function
  - Save/load index persistence
  - Document CRUD operations
  - Recursive reasoning over FAISS data

Prerequisites:
    pip install deeprecall[faiss]
    export OPENAI_API_KEY=sk-...
"""

import os
import shutil

from dotenv import load_dotenv
from openai import OpenAI

from deeprecall import DeepRecall
from deeprecall.vectorstores.faiss import FAISSStore

load_dotenv()

# --- Embedding function -------------------------------------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_fn(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


# --- Setup FAISS store --------------------------------------------------------
# FAISS runs entirely locally -- no server needed.
store = FAISSStore(dimension=1536, embedding_fn=embed_fn)

# --- Add documents ------------------------------------------------------------
print("Adding documents to FAISS index...")
ids = store.add_documents(
    documents=[
        "FAISS (Facebook AI Similarity Search) is a library for efficient "
        "similarity search and clustering of dense vectors, developed by Meta.",
        "FAISS supports GPU acceleration, enabling billion-scale nearest neighbor "
        "search in milliseconds using NVIDIA CUDA.",
        "The IVF (Inverted File) index in FAISS partitions the vector space into "
        "Voronoi cells for sub-linear search time.",
        "Product Quantization (PQ) in FAISS compresses vectors to reduce memory "
        "usage while maintaining search accuracy.",
        "FAISS is widely used in recommendation systems, image retrieval, and "
        "natural language processing for embedding search.",
        "The HNSW (Hierarchical Navigable Small World) index in FAISS provides "
        "excellent recall with logarithmic search complexity.",
    ],
    metadatas=[
        {"topic": "overview", "source": "meta"},
        {"topic": "gpu", "source": "meta"},
        {"topic": "ivf", "source": "research"},
        {"topic": "pq", "source": "research"},
        {"topic": "applications", "source": "industry"},
        {"topic": "hnsw", "source": "research"},
    ],
)
print(f"Added {len(ids)} documents, total count: {store.count()}")

# --- Search -------------------------------------------------------------------
print("\n--- Vector search ---")
results = store.search("How does FAISS handle large-scale search?", top_k=3)
for r in results:
    print(f"  Score: {r.score:.3f} | {r.content[:80]}...")

# --- Save and reload ----------------------------------------------------------
print("\n--- Saving index to disk ---")
save_dir = "./faiss_demo_index"
store.save(save_dir)
print(f"Index saved to {save_dir}/")

print("Reloading index from disk...")
loaded_store = FAISSStore.load(save_dir, embedding_fn=embed_fn)
print(f"Loaded store count: {loaded_store.count()}")

# Verify search works after reload
results = loaded_store.search("GPU acceleration", top_k=2)
if results:
    print(f"Search after reload: {results[0].content[:60]}...")
else:
    print("Search after reload: no results found")

# --- Query with DeepRecall engine ---------------------------------------------
print("\n--- DeepRecall recursive reasoning ---")
with DeepRecall(
    vectorstore=loaded_store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
) as engine:
    result = engine.query(
        "Explain the different indexing strategies in FAISS (IVF, PQ, HNSW) "
        "and when to use each one."
    )
    print(f"Answer: {result.answer[:300]}...")
    print(f"Sources: {len(result.sources)}, Time: {result.execution_time:.1f}s")

# --- Cleanup ------------------------------------------------------------------
store.close()
loaded_store.close()
shutil.rmtree(save_dir, ignore_errors=True)
print(f"\nDone! Cleaned up {save_dir}/")

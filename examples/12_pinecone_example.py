"""DeepRecall with Pinecone -- Managed vector database in the cloud.

Demonstrates:
  - PineconeStore setup with API key
  - Custom embedding function
  - Document CRUD operations
  - Recursive reasoning over Pinecone data

Prerequisites:
    pip install deeprecall[pinecone]
    export OPENAI_API_KEY=sk-...
    export PINECONE_API_KEY=pc-...

Note: Pinecone is a managed service. You need a Pinecone account and
an existing index. Create one at https://app.pinecone.io/
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from deeprecall import DeepRecall
from deeprecall.vectorstores.pinecone import PineconeStore

load_dotenv()

# --- Embedding function -------------------------------------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_fn(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


# --- Setup Pinecone store -----------------------------------------------------
# Make sure you've created an index in Pinecone dashboard first.
# Index must have dimension=1536 for text-embedding-3-small.
store = PineconeStore(
    index_name="deeprecall-demo",
    api_key=os.getenv("PINECONE_API_KEY"),
    dimension=1536,
    embedding_fn=embed_fn,
)

# --- Add documents ------------------------------------------------------------
print("Adding documents to Pinecone...")
ids = store.add_documents(
    documents=[
        "Amazon Web Services (AWS) launched in 2006 and dominates the cloud market "
        "with a 31% share as of 2025. Key services include EC2, S3, and Lambda.",
        "Google Cloud Platform (GCP) is known for its strength in data analytics "
        "(BigQuery), machine learning (Vertex AI), and Kubernetes (GKE).",
        "Microsoft Azure is the second-largest cloud provider, popular in enterprise "
        "environments due to tight integration with Microsoft 365 and Active Directory.",
        "DigitalOcean targets developers and small businesses with simple, affordable "
        "cloud computing. Known for Droplets (VMs) and App Platform.",
        "Cloudflare provides CDN, DDoS protection, and edge computing (Workers). "
        "Its network spans 300+ cities worldwide.",
    ],
    metadatas=[
        {"provider": "aws", "market_position": "leader"},
        {"provider": "gcp", "market_position": "challenger"},
        {"provider": "azure", "market_position": "leader"},
        {"provider": "digitalocean", "market_position": "niche"},
        {"provider": "cloudflare", "market_position": "edge"},
    ],
)
print(f"Added {len(ids)} documents")

# --- Search -------------------------------------------------------------------
print("\n--- Vector search ---")
results = store.search("best cloud for machine learning", top_k=3)
for r in results:
    print(f"  Score: {r.score:.3f} | {r.content[:80]}...")

# --- Query with DeepRecall engine ---------------------------------------------
print("\n--- DeepRecall recursive reasoning ---")
with DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
) as engine:
    result = engine.query(
        "Compare the top 3 cloud providers. What are each one's strengths "
        "and what types of companies should use each?"
    )
    print(f"Answer: {result.answer[:300]}...")
    print(f"Sources: {len(result.sources)}, Time: {result.execution_time:.1f}s")

# --- Cleanup ------------------------------------------------------------------
store.delete(ids)
print(f"\nCleaned up {len(ids)} documents from Pinecone.")

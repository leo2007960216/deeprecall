"""DeepRecall with Milvus -- Production-ready vector database example.

Prerequisites:
    pip install deeprecall[milvus]
    # Start Milvus: docker compose up -d (see https://milvus.io/docs/install_standalone-docker.md)
    export OPENAI_API_KEY=sk-...
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from deeprecall import DeepRecall
from deeprecall.vectorstores.milvus import MilvusStore

load_dotenv()

# Create an embedding function using OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_fn(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


# 1. Create Milvus store with embedding function
store = MilvusStore(
    collection_name="research_papers",
    uri="http://localhost:19530",
    dimension=1536,
    embedding_fn=embed_fn,
)

# 2. Add documents
store.add_documents(
    documents=[
        "Attention Is All You Need introduces the Transformer architecture, "
        "replacing recurrent layers with self-attention mechanisms.",
        "BERT uses bidirectional training of Transformer for language understanding, "
        "achieving state-of-the-art results on 11 NLP tasks.",
        "GPT-3 demonstrates that language models can be few-shot learners, "
        "achieving strong performance with minimal task-specific training.",
    ],
    metadatas=[
        {"title": "Attention Is All You Need", "year": 2017},
        {"title": "BERT", "year": 2018},
        {"title": "GPT-3", "year": 2020},
    ],
)

# 3. Query with recursive reasoning
engine = DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    verbose=True,
)

result = engine.query("How did the Transformer architecture evolve from 2017 to 2020?")
print(f"\nAnswer: {result.answer}")
print(f"Sources: {len(result.sources)}")

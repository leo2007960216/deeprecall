"""DeepRecall with LlamaIndex -- Use as a query engine.

Prerequisites:
    pip install deeprecall[chroma,llamaindex]
    export OPENAI_API_KEY=sk-...
"""

import os

from dotenv import load_dotenv

from deeprecall import DeepRecall
from deeprecall.adapters.llamaindex import DeepRecallQueryEngine
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()

# Setup
store = ChromaStore(collection_name="llamaindex_demo")
store.add_documents(
    documents=[
        "Climate change is causing rising sea levels at an accelerating rate.",
        "Renewable energy adoption grew by 25% globally in 2025.",
        "Carbon capture technology reached commercial viability in late 2024.",
        "The Paris Agreement targets a 1.5C limit on global warming.",
    ],
    metadatas=[
        {"topic": "sea_level"},
        {"topic": "renewable"},
        {"topic": "carbon_capture"},
        {"topic": "policy"},
    ],
)

engine = DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
)

# Use as a LlamaIndex query engine
query_engine = DeepRecallQueryEngine(engine=engine)
response = query_engine.query(
    "What progress has been made toward climate goals, and what challenges remain?"
)

print(f"Answer: {response}")
print(f"\nSources: {len(response.source_nodes)}")
for node in response.source_nodes:
    print(f"  - Score: {node.score:.3f} | {node.text[:80]}...")

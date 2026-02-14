"""DeepRecall Multi-Hop Reasoning Showcase.

Demonstrates DeepRecall's ability to answer complex questions that require
searching, reasoning, and searching again -- something standard RAG cannot do.

Prerequisites:
    pip install deeprecall[chroma]
    export OPENAI_API_KEY=sk-...
"""

import os

from dotenv import load_dotenv

from deeprecall import DeepRecall
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()

# Create a knowledge base that requires multi-hop reasoning
store = ChromaStore(collection_name="multi_hop_demo")
store.add_documents(
    documents=[
        # Company information (spread across documents)
        "TechCorp was founded in 2015 by Alice Chen. The company is headquartered in Austin, TX.",
        "TechCorp's main product is CloudSync, an enterprise data synchronization platform.",
        "CloudSync uses a proprietary algorithm called FastMerge for conflict resolution.",
        "In 2024, TechCorp acquired DataFlow Inc. for $200M to expand into real-time analytics.",
        "DataFlow's founder, Bob Smith, became TechCorp's VP of Engineering after the acquisition.",
        "TechCorp reported $500M revenue in 2025, with CloudSync accounting for 70%.",
        "The FastMerge algorithm was developed by Dr. Carol Zhang, TechCorp's Chief Scientist.",
        "TechCorp has 1,200 employees across offices in Austin, San Francisco, and London.",
        "In Q1 2026, TechCorp launched CloudSync 3.0 with AI-powered data reconciliation.",
        "CloudSync 3.0 reduced data sync conflicts by 95% compared to version 2.0.",
    ],
    metadatas=[
        {"category": "company", "year": 2015},
        {"category": "product", "year": 2020},
        {"category": "technology", "year": 2020},
        {"category": "acquisition", "year": 2024},
        {"category": "people", "year": 2024},
        {"category": "financials", "year": 2025},
        {"category": "people", "year": 2020},
        {"category": "company", "year": 2025},
        {"category": "product", "year": 2026},
        {"category": "product", "year": 2026},
    ],
)

engine = DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    verbose=True,
    max_iterations=10,
)

# Multi-hop question that requires connecting information across documents
print("=" * 70)
print("MULTI-HOP REASONING DEMO")
print("=" * 70)
print()

result = engine.query(
    "Who developed the core technology behind TechCorp's main product, "
    "and how has that product evolved since the DataFlow acquisition? "
    "Include revenue impact and technical improvements."
)

print("\n" + "=" * 70)
print("ANSWER:")
print("=" * 70)
print(result.answer)
print(f"\nSources accessed: {len(result.sources)}")
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Total LLM calls: {result.usage.total_calls}")
print(f"Total tokens: {result.usage.total_input_tokens + result.usage.total_output_tokens}")

"""DeepRecall Budget Guardrails -- Control how much a query can spend.

Demonstrates:
  - QueryBudget (search calls, tokens, time, cost limits)
  - Budget status inspection after a query
  - Reasoning trace walkthrough

Prerequisites:
    pip install deeprecall[chroma]
    export OPENAI_API_KEY=sk-...
"""

import os

from dotenv import load_dotenv

from deeprecall import DeepRecall, QueryBudget
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()

# --- Setup -------------------------------------------------------------------
store = ChromaStore(collection_name="budget_demo")
store.add_documents(
    documents=[
        "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. "
        "Elon Musk joined in 2004 as chairman and led its Series A funding.",
        "The Tesla Roadster (2008) was the first highway-legal all-electric car "
        "to use lithium-ion battery cells, achieving 245 miles of range.",
        "Model S launched in 2012 and was the world's best-selling plug-in "
        "electric car in 2015 and 2016.",
        "Tesla's Gigafactory in Nevada began production in 2016, targeting "
        "35 GWh/year of battery cell output by 2020.",
        "In 2020, Tesla became the most valuable automaker by market cap, "
        "surpassing Toyota. Revenue was $31.5B.",
        "Tesla FSD (Full Self-Driving) uses 8 cameras and a neural network "
        "vision system, replacing radar and ultrasonic sensors.",
        "Model 3 became the best-selling electric car globally in 2021, "
        "with cumulative sales exceeding 1 million units.",
        "Tesla's energy storage division deployed 6.5 GWh in 2022, a 64% "
        "year-over-year increase, primarily with Megapack installations.",
    ],
    metadatas=[
        {"topic": "founding", "year": 2003},
        {"topic": "roadster", "year": 2008},
        {"topic": "model_s", "year": 2012},
        {"topic": "gigafactory", "year": 2016},
        {"topic": "market_cap", "year": 2020},
        {"topic": "fsd", "year": 2021},
        {"topic": "model_3", "year": 2021},
        {"topic": "energy", "year": 2022},
    ],
)

# --- Example 1: Tight budget -------------------------------------------------
print("=" * 70)
print("EXAMPLE 1: Tight budget (max 3 searches, 20s timeout)")
print("=" * 70)

with DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
) as engine:
    result = engine.query(
        "Trace Tesla's journey from founding to becoming the most valuable automaker.",
        budget=QueryBudget(
            max_search_calls=3,
            max_tokens=30000,
            max_time_seconds=20.0,
        ),
    )

    print(f"\nAnswer: {result.answer[:200]}...")
    print(f"\nBudget status: {result.budget_status}")

# --- Example 2: Inspect the reasoning trace ----------------------------------
print("\n" + "=" * 70)
print("EXAMPLE 2: Walk through the reasoning trace")
print("=" * 70)

with DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    verbose=True,
) as engine:
    result = engine.query(
        "What is Tesla's FSD technology and how does it work?",
        budget=QueryBudget(max_search_calls=5),
    )

    print(f"\nAnswer: {result.answer}")
    print(f"\n--- Reasoning Trace ({len(result.reasoning_trace)} steps) ---")
    for step in result.reasoning_trace:
        print(f"\nStep {step.iteration}: {step.action}")
        if step.searches:
            for s in step.searches:
                print(f"  Searched: {s.get('query', 'N/A')}")
        if step.code:
            print(f"  Code: {step.code[:120]}...")
    print(f"\nConfidence: {result.confidence}")
    print(f"Total tokens: {result.usage.total_input_tokens + result.usage.total_output_tokens}")

"""DeepRecall Quickstart -- Minimal example with ChromaDB.

Prerequisites:
    pip install deeprecall[chroma]
    export OPENAI_API_KEY=sk-...
"""

import os

from dotenv import load_dotenv

from deeprecall import DeepRecall
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()

# 1. Create a vector store and add documents
store = ChromaStore(collection_name="quickstart_docs")
store.add_documents(
    documents=[
        "Python was created by Guido van Rossum and first released in 1991. "
        "It emphasizes code readability and supports multiple programming paradigms.",
        "Rust is a systems programming language focused on safety, speed, and concurrency. "
        "It was originally designed by Graydon Hoare at Mozilla Research.",
        "JavaScript was created by Brendan Eich in 1995 while he was at Netscape. "
        "It has become the dominant language for web development.",
        "Go (Golang) was designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson. "
        "It was announced in 2009 and is known for its simplicity and concurrency support.",
        "TypeScript is a superset of JavaScript developed by Microsoft. "
        "It adds static typing and was first released in 2012.",
    ],
    metadatas=[
        {"language": "Python", "year": 1991},
        {"language": "Rust", "year": 2010},
        {"language": "JavaScript", "year": 1995},
        {"language": "Go", "year": 2009},
        {"language": "TypeScript", "year": 2012},
    ],
)

# 2. Create the DeepRecall engine
engine = DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={
        "model_name": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    verbose=True,
)

# 3. Query with recursive reasoning
result = engine.query(
    "Compare the design philosophies of Python and Rust. "
    "Which one is better for systems programming and why?"
)

# 4. Print results
print("\n" + "=" * 60)
print("ANSWER:")
print("=" * 60)
print(result.answer)
print(f"\nSources: {len(result.sources)}")
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Total LLM calls: {result.usage.total_calls}")

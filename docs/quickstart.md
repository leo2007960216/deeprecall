# Quickstart

## Installation

```bash
# Core + ChromaDB (recommended for getting started)
pip install deeprecall[chroma]

# With all vector stores
pip install deeprecall[all]
```

## First Query

```python
import os
from deeprecall import DeepRecall
from deeprecall.vectorstores import ChromaStore

# Create a vector store
store = ChromaStore(collection_name="quickstart")

# Add documents
store.add_documents(
    documents=[
        "Python was created by Guido van Rossum in 1991.",
        "Rust focuses on memory safety without garbage collection.",
        "JavaScript is the language of the web.",
    ],
)

# Create engine
engine = DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={
        "model_name": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
)

# Query!
result = engine.query("Compare Python and Rust for systems programming.")
print(result.answer)
```

## What Happens Under the Hood

1. DeepRecall starts a local search server wrapping your vector store
2. It creates an RLM instance with a custom system prompt that knows about `search_db()`
3. The LLM enters a recursive reasoning loop where it can:
   - Call `search_db(query, top_k)` to search your documents
   - Call `llm_query(prompt)` to reason over retrieved content
   - Write Python code to analyze and transform data
4. When the LLM has enough information, it produces a `FINAL()` answer
5. DeepRecall returns the answer along with sources and usage stats

## Next Steps

- [Vector Stores](vectorstores.md) -- connect to your preferred database
- [Adapters](adapters.md) -- integrate with LangChain, LlamaIndex, or the OpenAI API
- [Architecture](architecture.md) -- understand how DeepRecall works

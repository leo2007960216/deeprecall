# DeepRecall Documentation

**DeepRecall** is a recursive reasoning engine that bridges AI agents with vector databases using MIT's [Recursive Language Models (RLM)](https://github.com/alexzhang13/rlm) paradigm.

## What Makes DeepRecall Different?

Standard RAG (Retrieval-Augmented Generation) performs a single pass: retrieve relevant documents, stuff them into a prompt, and generate an answer. DeepRecall replaces this with **recursive, multi-hop reasoning**:

1. The LLM decomposes your query into sub-questions
2. It searches the vector database for relevant documents
3. It writes and executes Python code to analyze retrieved results
4. It decides whether it needs more information and searches again
5. It synthesizes a comprehensive answer across all reasoning steps

## Getting Started

See [Quickstart](quickstart.md) for installation and first usage.

## Core Concepts

- **Engine**: The `DeepRecall` class orchestrates the recursive reasoning loop
- **Vector Stores**: Pluggable adapters for ChromaDB, Milvus, Qdrant, and Pinecone
- **Adapters**: Integrations for LangChain, LlamaIndex, and the OpenAI API
- **CLI**: Command-line tools for ingesting, querying, and serving

## Links

- [GitHub Repository](https://github.com/kothapavan/deeprecall)
- [PyPI Package](https://pypi.org/project/deeprecall/)
- [RLM Paper](https://arxiv.org/abs/2512.24601)

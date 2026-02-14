# Vector Stores

DeepRecall supports four vector database backends. Install only what you need.

## ChromaDB

Local-first, zero-config. Best for development and small datasets.

```bash
pip install deeprecall[chroma]
```

```python
from deeprecall.vectorstores import ChromaStore

# In-memory (default)
store = ChromaStore(collection_name="my_docs")

# Persistent storage
store = ChromaStore(collection_name="my_docs", persist_directory="./chroma_db")

# Client/server mode
store = ChromaStore(collection_name="my_docs", host="localhost", port=8000)
```

ChromaDB generates embeddings automatically using its built-in model.

## Milvus

Scalable, production-ready. Best for large datasets.

```bash
pip install deeprecall[milvus]
```

```python
from deeprecall.vectorstores import MilvusStore

store = MilvusStore(
    collection_name="my_docs",
    uri="http://localhost:19530",
    dimension=1536,  # Must match your embedding model
    embedding_fn=your_embedding_function,
)
```

Milvus requires an external embedding function.

## Qdrant

Fast, Rust-based. Best for performance-critical applications.

```bash
pip install deeprecall[qdrant]
```

```python
from deeprecall.vectorstores import QdrantStore

store = QdrantStore(
    collection_name="my_docs",
    url="http://localhost:6333",
    dimension=1536,
    embedding_fn=your_embedding_function,
)
```

## Pinecone

Managed cloud service. Best for zero-ops deployments.

```bash
pip install deeprecall[pinecone]
```

```python
from deeprecall.vectorstores import PineconeStore

store = PineconeStore(
    index_name="my-index",
    api_key="pc-...",
    dimension=1536,
    embedding_fn=your_embedding_function,
)
```

## Custom Vector Store

Implement `BaseVectorStore` to add your own:

```python
from deeprecall.vectorstores.base import BaseVectorStore

class MyStore(BaseVectorStore):
    def add_documents(self, documents, metadatas=None, ids=None, embeddings=None): ...
    def search(self, query, top_k=5, filters=None): ...
    def delete(self, ids): ...
    def count(self): ...
```

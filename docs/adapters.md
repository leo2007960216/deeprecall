# Framework Adapters

DeepRecall integrates with popular AI frameworks so you can drop it into existing pipelines.

## LangChain

```bash
pip install deeprecall[langchain]
```

### As a Retriever

```python
from deeprecall.adapters.langchain import DeepRecallRetriever

retriever = DeepRecallRetriever(engine=engine, top_k=10)
docs = retriever.invoke("Your question")
```

### As a Chat Model

```python
from deeprecall.adapters.langchain import DeepRecallChatModel

llm = DeepRecallChatModel(engine=engine)
response = llm.invoke("Your question")
print(response.content)
```

## LlamaIndex

```bash
pip install deeprecall[llamaindex]
```

### As a Query Engine

```python
from deeprecall.adapters.llamaindex import DeepRecallQueryEngine

query_engine = DeepRecallQueryEngine(engine=engine)
response = query_engine.query("Your question")
```

### As a Retriever

```python
from deeprecall.adapters.llamaindex import DeepRecallRetriever

retriever = DeepRecallRetriever(engine=engine, top_k=5)
nodes = retriever.retrieve("Your question")
```

## OpenAI-Compatible API

```bash
pip install deeprecall[server]
```

Start the server:

```bash
deeprecall serve --vectorstore chroma --collection my_docs --port 8000
```

Use with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="deeprecall",
    messages=[{"role": "user", "content": "Your question"}],
)
```

### Custom Endpoints

The server also provides:

- `POST /v1/documents` -- Add documents to the vector store
- `GET /health` -- Health check
- `GET /docs` -- Interactive API documentation (Swagger)

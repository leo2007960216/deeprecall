"""DeepRecall with LangChain -- Use as a retriever or chat model.

Prerequisites:
    pip install deeprecall[chroma,langchain]
    export OPENAI_API_KEY=sk-...
"""

import os

from dotenv import load_dotenv

from deeprecall import DeepRecall
from deeprecall.adapters.langchain import DeepRecallChatModel, DeepRecallRetriever
from deeprecall.vectorstores.chroma import ChromaStore

load_dotenv()

# Setup
store = ChromaStore(collection_name="langchain_demo")
store.add_documents(
    documents=[
        "The company's Q4 revenue was $2.5B, up 15% year-over-year.",
        "Operating expenses increased by 8% due to R&D investments.",
        "Net income was $450M, representing an 18% profit margin.",
        "The board approved a $500M stock buyback program.",
    ],
    metadatas=[
        {"section": "revenue"},
        {"section": "expenses"},
        {"section": "profit"},
        {"section": "shareholder"},
    ],
)

engine = DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
)

# Option 1: Use as a LangChain retriever
print("=" * 50)
print("Using as a LangChain Retriever:")
print("=" * 50)
retriever = DeepRecallRetriever(engine=engine)
docs = retriever.invoke("What was the company's financial performance?")
for doc in docs:
    print(f"  - {doc.page_content[:100]}...")

# Option 2: Use as a LangChain chat model
print("\n" + "=" * 50)
print("Using as a LangChain Chat Model:")
print("=" * 50)
llm = DeepRecallChatModel(engine=engine)
response = llm.invoke("Is the company profitable? What is the profit margin?")
print(f"  Answer: {response.content}")

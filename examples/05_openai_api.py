"""DeepRecall as an OpenAI-compatible API.

Prerequisites:
    pip install deeprecall[chroma,server]
    export OPENAI_API_KEY=sk-...

Step 1: Start the server (in a separate terminal):
    deeprecall serve --vectorstore chroma --collection api_demo --port 8000

Step 2: Run this script to query the server.
"""

from openai import OpenAI

# Connect to the DeepRecall server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# List available models
models = client.models.list()
print("Available models:")
for model in models.data:
    print(f"  - {model.id}")

# Query with recursive reasoning
print("\nQuerying DeepRecall...")
response = client.chat.completions.create(
    model="deeprecall",
    messages=[{"role": "user", "content": "What are the main themes in the documents?"}],
)

print(f"\nAnswer: {response.choices[0].message.content}")
print(f"Usage: {response.usage}")

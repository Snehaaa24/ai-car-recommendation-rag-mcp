# test_ollama.py

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:3b")

response = llm.invoke("What is FastAPI?")

print(response.content)
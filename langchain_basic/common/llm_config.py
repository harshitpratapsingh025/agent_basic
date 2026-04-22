"""
Shared LLM configuration — imported by all example files.
Change the model here and it applies everywhere.
"""

from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model="llama3.2:latest")

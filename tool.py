from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community import tools

text = "LangChain"

llm = ChatOllama(
    model="llama3.2",
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256,
)

sum = llm.invoke("What is capital of India?")

print(sum.content)
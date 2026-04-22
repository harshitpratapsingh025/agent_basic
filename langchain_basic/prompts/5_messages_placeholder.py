"""
================================================================================
Example 5: MessagesPlaceholder — Inject chat history into a prompt
================================================================================

WHAT:
  MessagesPlaceholder is a special placeholder that injects a LIST of
  messages into a ChatPromptTemplate. This is how you add memory/history.

HOW IT WORKS:
  1. Define a placeholder with variable_name="history".
  2. At invoke time, pass history=[HumanMessage(...), AIMessage(...)].
  3. LangChain inserts those messages between system and human messages.

WHY:
  Chat models perform much better when they can see the conversation
  context. MessagesPlaceholder is the clean way to inject that context
  into a ChatPromptTemplate without manually building message lists.

TRY THIS:
  - Add more messages to fake_history and see how the response changes.
  - Remove the history entirely and compare the response quality.

RUN:  python langchain/prompts/5_messages_placeholder.py
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_basic.common.llm_config import llm


def main():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful travel assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    # Simulate a conversation history
    fake_history = [
        HumanMessage(content="I want to visit Japan."),
        AIMessage(content="Great choice! Japan has amazing culture and food. "
                          "When are you planning to go?"),
        HumanMessage(content="Next spring, during cherry blossom season."),
        AIMessage(content="Perfect timing! Late March to mid-April is ideal."),
    ]

    print("\n--- LLM Response (with history context) ---")
    result = chain.invoke({
        "history": fake_history,
        "input": "What should I pack?",
    })
    print(result.content)


if __name__ == "__main__":
    main()

"""
================================================================================
Example 7: Interactive Chatbot with FileChatMessageHistory
================================================================================

WHAT:
  Build a stateful chatbot that remembers the conversation across turns.

HOW IT WORKS:
  1. FileChatMessageHistory stores messages in a local JSON file.
  2. Each turn:
       - Add HumanMessage to history
       - Invoke LLM with the full message list
       - Add AIMessage to history
  3. The LLM sees ALL previous messages, giving it conversational memory.

MESSAGE TYPES USED:
  - SystemMessage  → Sets the LLM's behavior ("You are a ...").
  - HumanMessage   → The user's input.
  - AIMessage      → The LLM's response.

NOTE:
  The history file (chat_history.json) persists between runs.
  Delete it to start a fresh conversation.

Type 'exit' to quit the chat loop.

RUN:  python langchain/prompts/7_chat_with_history.py
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_basic.common.llm_config import llm


def main():
    file_path = "./langchain/prompts/chat_history.json"
    history = FileChatMessageHistory(file_path)
    history.add_message(
        SystemMessage(content="You are a helpful assistant that helps plan vacations.")
    )

    print("\n--- Interactive Chat (type 'exit' to quit) ---")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        history.add_message(HumanMessage(content=query))
        result = llm.invoke(history.messages)
        history.add_message(AIMessage(content=result.content))
        print(f"AI: {result.content}\n")


if __name__ == "__main__":
    main()

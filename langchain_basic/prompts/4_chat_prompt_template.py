"""
================================================================================
Example 4: ChatPromptTemplate + LCEL Chain (pipe operator)
================================================================================

WHAT:
  ChatPromptTemplate is designed for chat models that expect messages
  with roles (system, human, ai).

HOW IT WORKS:
  1. from_messages() takes a list of (role, template_string) tuples.
  2. The pipe operator (|) chains the prompt with the LLM.
  3. chain.invoke({...}) formats the prompt AND calls the LLM in one step.

KEY INSIGHT — LCEL (LangChain Expression Language):
  Instead of calling .format() then llm.invoke() separately, you chain
  components with the | operator:

      chain = prompt | llm | output_parser

  This is the modern, recommended way to compose LangChain workflows.

ROLES:
  - "system"  → Sets the LLM's behavior/persona.
  - "human"   → The user's input (also accepts "user").
  - "ai"      → The assistant's response (also accepts "assistant").

TRY THIS:
  - Add ("ai", "...") messages to simulate prior conversation turns.
  - Change the system message to give the bot a different personality.

RUN:  python langchain/prompts/4_chat_prompt_template.py
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.prompts import ChatPromptTemplate
from langchain_basic.common.llm_config import llm


def main():
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "{user_input}"),
    ])

    # The | operator creates a Runnable chain: prompt → llm
    chain = chat_prompt | llm

    print("\n--- LLM Response ---")
    result = chain.invoke({"name": "Bob", "user_input": "What is your name?"})
    print(result.content)


if __name__ == "__main__":
    main()

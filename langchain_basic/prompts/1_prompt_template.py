"""
================================================================================
Example 1: PromptTemplate — Basic parameterized prompt
================================================================================

WHAT:
  PromptTemplate is the simplest and most common prompt type in LangChain.
  It works like Python's str.format() but with validation and serialization.

HOW IT WORKS:
  1. You write a template string with {placeholders}.
  2. You call .format() to fill them in → produces a plain string.
  3. You pass that string to llm.invoke().

KEY PARAMETERS:
  - template        : The string with {placeholders}.
  - input_variables : List of variable names the user MUST provide.
  - template_format : "f-string" (default) or "jinja2".

TRY THIS:
  - Change the business_name or business_description and re-run.
  - Try adding a new variable like {tone} to control the style.

RUN:  python langchain/prompts/1_prompt_template.py
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.prompts import PromptTemplate
from langchain_basic.common.llm_config import llm


def main():
    prompt_temp = PromptTemplate(
        template=(
            "You are an expert in marketing and brand strategy. "
            "My business name is {business_name} "
            "and my business description is {business_description}. "
            "Create a compelling brand slogan. "
            "Provide 3 options and explain the reasoning behind each."
        ),
        input_variables=["business_name", "business_description"],
    )

    # .format() fills in the variables and returns a plain string
    prompt_string = prompt_temp.format(
        business_name="ZenAI",
        business_description="AI-powered solutions for small businesses",
    )

    print("\n--- Formatted prompt (what the LLM actually sees) ---")
    print(prompt_string)
    print("\n--- LLM Response ---")
    result = llm.invoke(prompt_string)
    print(result.content)


if __name__ == "__main__":
    main()

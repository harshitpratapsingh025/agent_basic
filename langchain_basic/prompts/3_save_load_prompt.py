"""
================================================================================
Example 3: Save / Load PromptTemplate from JSON
================================================================================

WHAT:
  PromptTemplates can be serialized to JSON (or YAML) and loaded back.

WHY:
  - Version-control your prompts in git.
  - Share prompts across projects or teams.
  - Let non-developers edit prompts without touching Python code.

HOW IT WORKS:
  - prompt.save("template.json")              → writes to disk
  - PromptTemplate.from_file("template.json") → reads it back

  The file template.json in this folder was created with .save().

TRY THIS:
  - Open template.json and edit the template text, then re-run this script.
  - Create a new template, save it, and load it back.

RUN:  python langchain/prompts/3_save_load_prompt.py
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.prompts import PromptTemplate
from langchain_basic.common.llm_config import llm


def main():
    loaded_prompt = PromptTemplate.from_file("./langchain/prompts/template.json")

    print("\n--- Loaded template variables ---")
    print(f"  input_variables: {loaded_prompt.input_variables}")
    print(f"  template_format: {loaded_prompt.template_format}")

    prompt_string = loaded_prompt.format(
        business_name="ZenAI",
        business_description="AI-powered solutions for small businesses",
    )

    print("\n--- LLM Response ---")
    result = llm.invoke(prompt_string)
    print(result.content)


if __name__ == "__main__":
    main()

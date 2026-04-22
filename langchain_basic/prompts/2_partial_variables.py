"""
================================================================================
Example 2: Partial Variables — Pre-fill some placeholders
================================================================================

WHAT:
  partial_variables let you lock in some values at template-creation time.
  The caller only needs to supply the remaining variables.

USE CASE:
  - A variable that's always the same (e.g. company name).
  - A variable computed at runtime (e.g. today's date via a lambda).

HOW IT WORKS:
  - Pass partial_variables={"key": "value"} when creating the template.
  - Those keys are removed from input_variables — the user doesn't need
    to supply them.
  - You can also pass a callable (lambda) that runs at .format() time.

TRY THIS:
  - Uncomment the lambda version of "date" to see dynamic partial variables.
  - Add another partial variable like "language": "English".

RUN:  python langchain/prompts/2_partial_variables.py
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.prompts import PromptTemplate
from langchain_basic.common.llm_config import llm


def main():
    prompt = PromptTemplate(
        template=(
            "Company: {company}. "
            "Today's date: {date}. "
            "Write a short marketing email about {topic}."
        ),
        input_variables=["topic"],  # only topic needs to be supplied
        partial_variables={
            "company": "ZenAI",
            "date": "2026-03-31",
            # "date": lambda: datetime.now().strftime("%Y-%m-%d"),  # dynamic!
        },
    )

    # Only need to provide 'topic' — the rest is pre-filled
    prompt_string = prompt.format(topic="our new AI dashboard")

    print("\n--- Formatted prompt ---")
    print(prompt_string)
    print("\n--- LLM Response ---")
    result = llm.invoke(prompt_string)
    print(result.content)


if __name__ == "__main__":
    main()

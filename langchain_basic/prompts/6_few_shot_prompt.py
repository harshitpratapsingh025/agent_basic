"""
================================================================================
Example 6: FewShotPromptTemplate — Teach the LLM by example
================================================================================

WHAT:
  FewShotPromptTemplate includes example input/output pairs so the LLM
  learns the desired format. This is called "in-context learning" or
  "few-shot prompting".

HOW IT WORKS:
  1. Define examples as a list of dicts.
  2. Define an example_prompt that formats each individual example.
  3. FewShotPromptTemplate combines:
       prefix + formatted examples + suffix + your input

STRUCTURE OF THE FINAL PROMPT:
  [prefix]
  Input: happy          ← example 1
  Output: sad
  Input: tall           ← example 2
  Output: short
  [suffix]
  Input: bright         ← your actual input
  Output:               ← LLM fills this in

TRY THIS:
  - Add more examples and see if accuracy improves.
  - Change the task (e.g. synonyms instead of antonyms).
  - Try with just 1 example vs 5 examples — notice the difference.

RUN:  python langchain/prompts/6_few_shot_prompt.py
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_basic.common.llm_config import llm


def main():
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "fast", "output": "slow"},
    ]

    example_prompt = PromptTemplate(
        template="Input: {input}\nOutput: {output}",
        input_variables=["input", "output"],
    )

    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input.\n",
        suffix="\nInput: {input}\nOutput:",
        input_variables=["input"],
    )

    prompt_string = few_shot.format(input="bright")

    print("\n--- Formatted few-shot prompt ---")
    print(prompt_string)
    print("\n--- LLM Response ---")
    result = llm.invoke(prompt_string)
    print(result.content)


if __name__ == "__main__":
    main()

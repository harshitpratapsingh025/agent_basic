import sys
import os

# Add project root to path so imports work when running the file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.runnables import RunnableParallel
from langchain_basic.common.llm_config import llm
from langchain_core.prompts import PromptTemplate

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
print(result)


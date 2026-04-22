"""
This file has been split into individual example files for easier learning.

See the files in this folder:

  llm_config.py             → Shared LLM setup (change model here)
  1_prompt_template.py      → Basic PromptTemplate with {placeholders}
  2_partial_variables.py    → Pre-fill variables at template creation time
  3_save_load_prompt.py     → Serialize / deserialize prompts to JSON
  4_chat_prompt_template.py → ChatPromptTemplate + LCEL pipe operator
  5_messages_placeholder.py → Inject chat history into a prompt
  6_few_shot_prompt.py      → Teach the LLM by providing examples
  7_chat_with_history.py    → Interactive chatbot with persistent memory

  run_all.py                → Menu to pick and run any example

Run any example directly:
  python langchain/prompts/1_prompt_template.py

Or use the menu:
  python langchain/prompts/run_all.py
"""

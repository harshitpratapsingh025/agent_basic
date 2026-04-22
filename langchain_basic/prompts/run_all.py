"""
================================================================================
LangChain Prompts — Interactive Example Runner
================================================================================

Run this file to pick which example to study:

  python langchain/prompts/run_all.py

Or run any example directly:

  python langchain/prompts/1_prompt_template.py
  python langchain/prompts/2_partial_variables.py
  ...etc

Files:
  llm_config.py             → Shared LLM setup (change model here)
  1_prompt_template.py      → Basic PromptTemplate with {placeholders}
  2_partial_variables.py    → Pre-fill variables at template creation time
  3_save_load_prompt.py     → Serialize / deserialize prompts to JSON
  4_chat_prompt_template.py → ChatPromptTemplate + LCEL pipe operator
  5_messages_placeholder.py → Inject chat history into a prompt
  6_few_shot_prompt.py      → Teach the LLM by providing examples
  7_chat_with_history.py    → Interactive chatbot with persistent memory
================================================================================
"""

import importlib.util
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EXAMPLES = {
    "1": ("PromptTemplate — Basic parameterized prompt", "1_prompt_template.py"),
    "2": ("Partial Variables — Pre-fill placeholders", "2_partial_variables.py"),
    "3": ("Save / Load — PromptTemplate from JSON", "3_save_load_prompt.py"),
    "4": ("ChatPromptTemplate + LCEL Chain", "4_chat_prompt_template.py"),
    "5": ("MessagesPlaceholder — Inject chat history", "5_messages_placeholder.py"),
    "6": ("FewShotPromptTemplate — Teach by example", "6_few_shot_prompt.py"),
    "7": ("Interactive Chatbot with memory", "7_chat_with_history.py"),
}


def run_example(filename):
    """Dynamically import and run a numbered example file."""
    filepath = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main():
    print("\n" + "=" * 60)
    print("  LangChain Prompts — Interactive Examples")
    print("=" * 60)
    for key, (desc, _) in EXAMPLES.items():
        print(f"  {key}. {desc}")
    print("  a. Run ALL examples (except interactive chat)")
    print("  q. Quit")
    print("=" * 60)

    choice = input("\nPick an example to run: ").strip().lower()

    if choice == "q":
        return
    elif choice == "a":
        for key, (desc, filename) in EXAMPLES.items():
            if key == "7":
                continue
            print(f"\n{'=' * 60}")
            print(f"  Running: {desc}")
            print(f"{'=' * 60}")
            run_example(filename)
    elif choice in EXAMPLES:
        run_example(EXAMPLES[choice][1])
    else:
        print("Invalid choice. Run the script again.")


if __name__ == "__main__":
    main()

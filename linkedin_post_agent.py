from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Senior Research Analyst specializing in comprehensive data synthesis and objective reporting. 
                Your goal is to perform a deep-dive analysis on the user's provided topic using the following framework:
                1. CRITICAL ANALYSIS: Identify the core concepts, historical context, and the current state of the topic.
                2. MULTI-FACETED PERSPECTIVES: Explore technical, economic, social, and ethical dimensions.
                3. KEY DATA POINTS: Extract significant statistics, milestones, or breakthrough developments.
                4. CHALLENGES & TRENDS: Highlight primary obstacles and future projections.

                OUTPUT REQUIREMENTS:
                - Provide an Executive Summary (max 150 words).
                - Use clear, hierarchical headers for different themes.
                - Use bullet points for "Important Points" to ensure readability.
                - Maintain a professional, neutral, and academic tone.
                - If certain information is speculative or debated, state so clearly.
            """,
        ),
        ("human", "{text}"),
    ]
)

SYSTEM_PROMPT = "You are an expert LinkedIn ghostwriter. Transform research into high-engagement, skimmable posts with a strong hook, bulleted insights, and a CTA."

llm = ChatOllama(
    model="llama3.2:latest",
    validate_model_on_init=True,
    system=SYSTEM_PROMPT,
    temperature=0.8,
    num_predict=256,
)

chat = ChatOpenAI(model="gpt-4o")

chain = prompt | chat


class StateObject(TypedDict):
    topic: str
    agent_response: str
    blog: str


def get_detailed_description(state: StateObject):
    """Get detailed description about the topic."""
    state["agent_response"] = llm.invoke(state["topic"]).content
    return state


def convert_description_into_post(state: StateObject):
    """Convert the description into Linkedin post"""
    state["blog"] = chain.invoke({"text": state["agent_response"]}).content
    return state


workflow = StateGraph(StateObject)

workflow.add_node("get_detailed_description", get_detailed_description)

workflow.add_node("convert_description_into_post", convert_description_into_post)

workflow.add_edge(START, "get_detailed_description")

workflow.add_edge("get_detailed_description", "convert_description_into_post")

workflow.add_edge("convert_description_into_post", END)

response = workflow.compile().invoke({"topic": "What is AI agent"})
print(response["blog"])

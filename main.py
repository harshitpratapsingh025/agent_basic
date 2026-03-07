from dotenv import load_dotenv
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.agents.structured_output import ToolStrategy


load_dotenv()

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
     response_format=ToolStrategy(ResponseFormat),
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response['structured_response'])
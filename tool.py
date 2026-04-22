from datetime import datetime
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, PIIMiddleware
from langchain.agents.structured_output import ToolStrategy


load_dotenv()

# We use a dataclass here, but Pydantic models are also supported.
class Geo(BaseModel):
    lat: str
    lng: str


class Address(BaseModel):
    street: str
    suite: str
    city: str
    zipcode: str
    geo: Geo


class Company(BaseModel):
    name: str
    catchPhrase: str
    bs: str


class UserDetails(BaseModel):
    id: int
    name: str
    username: str
    email: str
    address: Address
    phone: str
    website: str
    company: Company


class Post(BaseModel):
    userId: int
    id: int
    title: str
    body: str

class User(BaseModel):
    id: int
    name: str = "John Doe"
    username: str
    email: str
    posts: list[Post]


def get_user_details(name: str) -> UserDetails:
    """Get user details from user name."""
    response = requests.get("https://jsonplaceholder.typicode.com/users")
    users: list[UserDetails] = response.json()
    user = next((item for item in users if item["name"] == name), None)
    return user


def get_user_post_by_id(id: str) -> list[Post]:
    """Get user all posts using user id."""
    response = requests.get(f"https://jsonplaceholder.typicode.com/posts?userId={id}")
    posts: list[Post] = response.json()
    return posts


agent = create_agent(
    model="gpt-4o",
    tools=[get_user_details, get_user_post_by_id],
    middleware=[
        ToolCallLimitMiddleware(thread_limit=20, run_limit=5),
        PIIMiddleware("email", strategy="mask"),
    ],
    system_prompt="You are a helpful assistant. Your task is first get the user details using user name then find all the user posts using user id and return a structure response user details with all post lists",
    response_format=ToolStrategy(User),
)

# Run the agent
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Give me full details of user Clementine Bauch"}
        ]
    }
)

print(dict(response["structured_response"]))

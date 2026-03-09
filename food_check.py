from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph


class UserFoodCheckRequest(BaseModel):
    dish_name: str
    diseases: List[str]


class Ingredient:
    id: str
    name: str
    category: str
    nutrition: Dict
    
class Dish:
    name: str
    cuisine: str
    ingredients: List[str]
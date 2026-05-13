from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict


class State(TypedDict):
    query: str
    intent: str
    context: list[str]
    messages: Annotated[list[BaseMessage], add_messages]
    output: str
    user_preferences: dict
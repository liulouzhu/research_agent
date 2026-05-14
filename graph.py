import os

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agents.classifier import classify_intent, route_by_intent
from agents.polish import polish_agent
from agents.qa import qa_agent
from agents.writing import writing_agent
from state import State


def _make_classifier_node(llm):
    def node(state: State) -> dict:
        return classify_intent(state, llm)
    return node


def _make_qa_node(llm, chroma_dir):
    def node(state: State) -> dict:
        return qa_agent(state, llm, chroma_dir)
    return node


def _make_writing_node(llm):
    def node(state: State) -> dict:
        return writing_agent(state, llm)
    return node


def _make_polish_node(llm):
    def node(state: State) -> dict:
        return polish_agent(state, llm)
    return node


def build_graph(chroma_dir: str = "./chroma_db", model: str = None):
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    graph = StateGraph(State)

    graph.add_node("classifier", _make_classifier_node(llm))
    graph.add_node("qa_agent", _make_qa_node(llm, chroma_dir))
    graph.add_node("writing_agent", _make_writing_node(llm))
    graph.add_node("polish_agent", _make_polish_node(llm))

    graph.set_entry_point("classifier")

    graph.add_conditional_edges("classifier", route_by_intent, {
        "qa_agent": "qa_agent",
        "writing_agent": "writing_agent",
        "polish_agent": "polish_agent",
    })

    graph.add_edge("qa_agent", END)
    graph.add_edge("writing_agent", END)
    graph.add_edge("polish_agent", END)

    return graph.compile()
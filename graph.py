import os

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agents.classifier import classify_intent, route_by_intent
from agents.arxiv import arxiv_agent
from agents.general import general_agent
from agents.innovation import innovation_agent
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


def _make_general_node(llm):
    def node(state: State) -> dict:
        return general_agent(state, llm)
    return node


def _make_arxiv_node(llm, chroma_dir):
    def node(state: State) -> dict:
        return arxiv_agent(state, llm, chroma_dir)
    return node


def _make_innovation_node(llm, reviewer_llm, chroma_dir):
    def node(state: State) -> dict:
        return innovation_agent(state, llm, reviewer_llm=reviewer_llm, chroma_dir=chroma_dir)
    return node


def build_graph(chroma_dir: str = "./chroma_db", model: str = None, reviewer_model: str = None):
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    reviewer_llm = None
    if reviewer_model:
        reviewer_llm = ChatOpenAI(model=reviewer_model, temperature=0)
    elif os.environ.get("REVIEWER_MODEL"):
        reviewer_llm = ChatOpenAI(model=os.environ["REVIEWER_MODEL"], temperature=0)

    graph = StateGraph(State)

    graph.add_node("classifier", _make_classifier_node(llm))
    graph.add_node("qa_agent", _make_qa_node(llm, chroma_dir))
    graph.add_node("writing_agent", _make_writing_node(llm))
    graph.add_node("polish_agent", _make_polish_node(llm))
    graph.add_node("general_agent", _make_general_node(llm))
    graph.add_node("arxiv_agent", _make_arxiv_node(llm, chroma_dir))
    graph.add_node("innovation_agent", _make_innovation_node(llm, reviewer_llm, chroma_dir))

    graph.set_entry_point("classifier")

    graph.add_conditional_edges("classifier", route_by_intent, {
        "qa_agent": "qa_agent",
        "writing_agent": "writing_agent",
        "polish_agent": "polish_agent",
        "general_agent": "general_agent",
        "arxiv_agent": "arxiv_agent",
        "innovation_agent": "innovation_agent",
    })

    graph.add_edge("qa_agent", END)
    graph.add_edge("writing_agent", END)
    graph.add_edge("polish_agent", END)
    graph.add_edge("general_agent", END)
    graph.add_edge("arxiv_agent", END)
    graph.add_edge("innovation_agent", END)

    return graph.compile()
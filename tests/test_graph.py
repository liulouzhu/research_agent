from unittest.mock import MagicMock, patch
from graph import build_graph


@patch("graph.ChatOpenAI")
def test_build_graph_returns_compiled(mock_llm_cls):
    mock_llm = MagicMock()
    mock_llm_cls.return_value = mock_llm
    graph = build_graph()
    assert graph is not None


@patch("rag.retriever.OpenAIEmbeddings")
@patch("rag.retriever.Chroma")
@patch("graph.ChatOpenAI")
def test_graph_routes_qa_flow(mock_llm_cls, mock_chroma_cls, mock_emb):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    mock_llm = MagicMock()
    call_count = [0]

    def invoke_side_effect(msgs):
        call_count[0] += 1
        if call_count[0] == 1:
            return AIMessage(content="qa")
        return AIMessage(content="注意力机制是一种神经网络技术。")

    mock_llm.invoke.side_effect = invoke_side_effect
    mock_llm_cls.return_value = mock_llm

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [
        (MagicMock(page_content="Attention is a neural network technique.", metadata={"source": "paper.pdf", "page": 0}), 0.1),
    ]
    mock_chroma_cls.return_value = mock_vs

    graph = build_graph()
    result = graph.invoke({
        "query": "什么是注意力机制",
        "intent": "",
        "context": [],
        "messages": [],
        "output": "",
        "user_preferences": {},
    })
    assert "注意力" in result["output"]
    assert call_count[0] == 2
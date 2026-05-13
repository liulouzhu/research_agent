from unittest.mock import MagicMock, patch

from agents.qa import qa_agent


@patch("agents.qa.retrieve")
def test_qa_agent_with_results(mock_retrieve):
    mock_retrieve.return_value = [
        {"content": "Attention is all you need.", "source": "paper.pdf", "page": 0, "score": 0.1}
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="注意力机制是一种神经网络技术。来源：paper.pdf 第0页")

    state = {
        "query": "什么是注意力机制",
        "messages": [],
        "context": [],
        "user_preferences": {"language": "chinese", "style": "academic"},
    }
    result = qa_agent(state, mock_llm, "./chroma_db")
    assert "注意力" in result["output"]
    assert len(result["context"]) == 1


@patch("agents.qa.retrieve")
def test_qa_agent_empty_results(mock_retrieve):
    mock_retrieve.return_value = []
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="未找到相关文档，无法回答该问题。")

    state = {
        "query": "something obscure",
        "messages": [],
        "context": [],
        "user_preferences": {},
    }
    result = qa_agent(state, mock_llm, "./chroma_db")
    assert "未找到" in result["output"]
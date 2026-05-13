from unittest.mock import MagicMock

from agents.writing import writing_agent


def test_writing_agent_generates_content():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Transformer是一种基于自注意力机制的模型架构...")

    state = {
        "query": "帮我写一段关于Transformer的介绍",
        "messages": [],
        "context": [],
        "user_preferences": {"language": "chinese", "style": "academic"},
    }
    result = writing_agent(state, mock_llm)
    assert "Transformer" in result["output"]


def test_writing_agent_with_history():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="续写内容...")

    from langchain_core.messages import HumanMessage, AIMessage
    state = {
        "query": "继续写",
        "messages": [HumanMessage(content="帮我写关于RNN的内容"), AIMessage(content="RNN是一种...")],
        "context": [],
        "user_preferences": {},
    }
    result = writing_agent(state, mock_llm)
    assert result["output"] is not None

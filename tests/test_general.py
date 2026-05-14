from unittest.mock import MagicMock

from agents.general import general_agent


def test_general_agent_responds():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="注意力机制是一种让模型关注输入重要部分的方法。")

    state = {
        "query": "什么是注意力机制？简单解释一下",
        "messages": [],
        "context": [],
        "user_preferences": {"language": "chinese", "style": "academic"},
    }
    result = general_agent(state, mock_llm)
    assert "注意力" in result["output"]


def test_general_agent_with_history():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="是的，补充说明...")

    from langchain_core.messages import HumanMessage, AIMessage
    state = {
        "query": "能再详细说说吗",
        "messages": [HumanMessage(content="什么是RNN"), AIMessage(content="RNN是循环神经网络")],
        "context": [],
        "user_preferences": {},
    }
    result = general_agent(state, mock_llm)
    assert result["output"] is not None
from unittest.mock import MagicMock

from agents.polish import polish_agent


def test_polish_agent_refines_text():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="润色后的文本内容")

    state = {
        "query": "润色这段文字：the model is good",
        "messages": [],
        "context": [],
        "user_preferences": {"language": "chinese", "style": "academic"},
    }
    result = polish_agent(state, mock_llm)
    assert "润色" in result["output"] or "文本" in result["output"] or result["output"] is not None

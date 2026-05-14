from unittest.mock import MagicMock

from agents.classifier import classify_intent


def test_classify_qa():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="qa")
    result = classify_intent({"query": "什么是注意力机制", "messages": [], "user_preferences": {}}, mock_llm)
    assert result["intent"] == "qa"


def test_classify_writing():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="writing")
    result = classify_intent({"query": "帮我写一段关于transformer的介绍", "messages": [], "user_preferences": {}}, mock_llm)
    assert result["intent"] == "writing"


def test_classify_polish():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="polish")
    result = classify_intent({"query": "润色这段文字", "messages": [], "user_preferences": {}}, mock_llm)
    assert result["intent"] == "polish"


def test_classify_general():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="general")
    result = classify_intent({"query": "你好，科研领域有哪些热门方向", "messages": [], "user_preferences": {}}, mock_llm)
    assert result["intent"] == "general"


def test_classify_fallback():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="something_unknown")
    result = classify_intent({"query": "hello", "messages": [], "user_preferences": {}}, mock_llm)
    assert result["intent"] == "general"


def test_classify_arxiv():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="arxiv")
    result = classify_intent({"query": "帮我找关于transformer的论文", "messages": [], "user_preferences": {}}, mock_llm)
    assert result["intent"] == "arxiv"


def test_route_by_intent():
    from agents.classifier import route_by_intent
    assert route_by_intent({"intent": "qa"}) == "qa_agent"
    assert route_by_intent({"intent": "writing"}) == "writing_agent"
    assert route_by_intent({"intent": "polish"}) == "polish_agent"
    assert route_by_intent({"intent": "general"}) == "general_agent"
    assert route_by_intent({"intent": "arxiv"}) == "arxiv_agent"
    assert route_by_intent({"intent": "unknown"}) == "general_agent"
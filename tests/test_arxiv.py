# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch

from agents.arxiv import arxiv_agent, search_arxiv


@patch("agents.arxiv.arxiv.Client")
def test_search_arxiv_returns_results(mock_client_cls):
    mock_client = MagicMock()
    mock_paper = MagicMock()
    mock_paper.title = "Attention Is All You Need"
    mock_author = MagicMock()
    mock_author.name = "Vaswani"
    mock_paper.authors = [mock_author]
    mock_paper.summary = "A novel architecture for sequence modeling."
    mock_paper.pdf_url = "https://arxiv.org/pdf/1706.03762"
    mock_paper.entry_id = "http://arxiv.org/abs/1706.03762v5"
    mock_paper.published = None
    mock_client.results.return_value = [mock_paper]
    mock_client_cls.return_value = mock_client

    results = search_arxiv("attention mechanism")
    assert len(results) == 1
    assert results[0]["title"] == "Attention Is All You Need"


@patch("agents.arxiv.search_arxiv")
def test_arxiv_agent_with_results(mock_search):
    mock_search.return_value = [
        {
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "summary": "A novel architecture...",
            "pdf_url": "https://arxiv.org/pdf/1706.03762",
            "entry_id": "http://arxiv.org/abs/1706.03762v5",
            "published": "2017-06-12",
        }
    ]
    mock_llm = MagicMock()

    state = {
        "query": "find papers about attention",
        "messages": [],
        "context": [],
        "user_preferences": {"language": "chinese"},
    }
    result = arxiv_agent(state, mock_llm, "./chroma_db")
    assert "Attention" in result["output"]


@patch("agents.arxiv.search_arxiv")
def test_arxiv_agent_no_results(mock_search):
    mock_search.return_value = []
    mock_llm = MagicMock()

    state = {
        "query": "obscure topic",
        "messages": [],
        "context": [],
        "user_preferences": {"language": "chinese"},
    }
    result = arxiv_agent(state, mock_llm, "./chroma_db")
    assert "arXiv" in result["output"]
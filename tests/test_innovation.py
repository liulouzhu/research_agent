# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch

from agents.innovation import innovation_agent, _generate_candidates, _analyze_novelty, _review_candidates, _parse_json_robust


def test_parse_json_robust_plain():
    result = _parse_json_robust('[{"title": "test"}]')
    assert len(result) == 1
    assert result[0]["title"] == "test"


def test_parse_json_robust_markdown():
    result = _parse_json_robust('```json\n[{"title": "test"}]\n```')
    assert len(result) == 1


def test_parse_json_robust_with_surrounding_text():
    result = _parse_json_robust('Here is the result:\n[{"title": "test"}]\nDone.')
    assert len(result) == 1


@patch("agents.innovation.search_arxiv")
def test_innovation_agent_with_results(mock_search):
    mock_search.return_value = [
        {
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "summary": "A novel architecture.",
            "pdf_url": "https://arxiv.org/pdf/1706.03762",
            "entry_id": "http://arxiv.org/abs/1706.03762v5",
            "published": "2017-06-12",
        }
    ]
    mock_llm = MagicMock()
    call_count = [0]

    def side_effect(msgs):
        call_count[0] += 1
        if call_count[0] == 1:
            return MagicMock(content='[{"title": "Dynamic Attention Routing", "problem": "Static attention is limiting", "idea": "Learn attention patterns dynamically", "key_difference": "Uses learned routing vs fixed patterns", "expected_value": "Better efficiency"}]')
        if call_count[0] == 2:
            return MagicMock(content='{"novelty_level": "HIGH", "overlap_dimensions": [], "similarities": [], "differences": ["Novel routing mechanism"], "novelty_conclusion": "High novelty"}')
        if call_count[0] == 3:
            return MagicMock(content='{"reviews": [{"title": "Dynamic Attention Routing", "novelty_score": 8, "feasibility_score": 7, "impact_score": 8, "overall_score": 7.7, "recommendation": "PROCEED", "comment": "Promising"}], "top_recommendations": ["Dynamic Attention Routing"]}')
        return MagicMock(content="{}")

    mock_llm.invoke.side_effect = side_effect

    state = {
        "query": "transformer attention mechanism",
        "messages": [],
        "context": [],
        "user_preferences": {"language": "english"},
    }
    result = innovation_agent(state, mock_llm)
    assert "Dynamic Attention Routing" in result["output"]
    assert call_count[0] >= 3


@patch("agents.innovation.search_arxiv")
def test_innovation_agent_no_candidates(mock_search):
    mock_search.return_value = []
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="I cannot generate candidates.")

    state = {
        "query": "obscure topic",
        "messages": [],
        "context": [],
        "user_preferences": {"language": "chinese"},
    }
    result = innovation_agent(state, mock_llm)
    assert result["output"] is not None
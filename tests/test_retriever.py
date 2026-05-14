import pytest
from unittest.mock import MagicMock, patch


@patch("rag.ingest.OpenAIEmbeddings")
def test_retrieve_returns_context(mock_embeddings_cls):
    mock_emb = MagicMock()
    mock_embeddings_cls.return_value = mock_emb
    mock_collection = MagicMock()
    mock_collection.similarity_search_with_score.return_value = [
        (MagicMock(page_content="Attention is all you need.", metadata={"source": "paper.pdf", "page": 0}), 0.1),
    ]

    with patch("rag.retriever.Chroma", return_value=mock_collection):
        from rag.retriever import retrieve
        results = retrieve("what is attention", "./chroma_db")
    assert len(results) == 1
    assert "Attention is all you need" in results[0]["content"]
    assert results[0]["source"] == "paper.pdf"
    assert results[0]["page"] == 0


@patch("rag.ingest.OpenAIEmbeddings")
def test_retrieve_empty(mock_embeddings_cls):
    mock_emb = MagicMock()
    mock_embeddings_cls.return_value = mock_emb
    mock_collection = MagicMock()
    mock_collection.similarity_search_with_score.return_value = []

    with patch("rag.retriever.Chroma", return_value=mock_collection):
        from rag.retriever import retrieve
        results = retrieve("obscure query", "./chroma_db")
    assert len(results) == 0
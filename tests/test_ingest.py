import fitz
import os
import pytest
from rag.ingest import ingest_document


@pytest.fixture
def sample_pdf(tmp_path):
    path = str(tmp_path / "sample.pdf")
    doc = fitz.open()
    page = doc.new_page()
    text = "Attention mechanism is a technique in neural networks. It allows the model to focus on specific parts of the input."
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def chroma_dir(tmp_path):
    return str(tmp_path / "chroma_db")


def test_ingest_creates_collection(sample_pdf, chroma_dir):
    n_chunks = ingest_document(sample_pdf, chroma_dir)
    assert n_chunks > 0


def test_ingest_pdf_not_found(chroma_dir):
    with pytest.raises(FileNotFoundError):
        ingest_document("/nonexistent.pdf", chroma_dir)
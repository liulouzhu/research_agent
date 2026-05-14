# -*- coding: utf-8 -*-
import os

import pytest

from rag.ingest import _text_to_markdown


def test_text_to_markdown_with_numbered_section():
    text = "1. Introduction\nThis is the intro.\n2. Related Work\nSome related work here."
    md = _text_to_markdown(text)
    assert "## 1. Introduction" in md
    assert "## 2. Related Work" in md


def test_text_to_markdown_with_subsection():
    text = "2.1 Subsection Title\nSome content."
    md = _text_to_markdown(text)
    assert "### 2.1 Subsection Title" in md


def test_text_to_markdown_with_abstract():
    text = "Abstract\nThis is the abstract.\n1. Introduction\nIntro content."
    md = _text_to_markdown(text)
    assert "# Abstract" in md
    assert "# 1. Introduction" in md


def test_text_to_markdown_preserves_plain_text():
    text = "Some plain text without headers.\nMore text."
    md = _text_to_markdown(text)
    assert "Some plain text without headers." in md


def test_text_to_markdown_with_keyword_header():
    text = "Conclusion\nWe conclude that...\nReferences\n[1] Paper A"
    md = _text_to_markdown(text)
    assert "# Conclusion" in md
    assert "# References" in md


def test_text_to_markdown_deep_subsection():
    text = "3.2.1 Details\nDetailed content."
    md = _text_to_markdown(text)
    assert "### 3.2.1 Details" in md


def test_ingest_pdf_not_found():
    from rag.ingest import ingest_document
    with pytest.raises(FileNotFoundError):
        ingest_document("/nonexistent.pdf", "./chroma_db")
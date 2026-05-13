import os

import fitz
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _extract_text(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(
                {"text": text, "metadata": {"source": os.path.basename(pdf_path), "page": page_num}}
            )
    doc.close()
    return pages


def ingest_document(pdf_path: str, chroma_dir: str = "./chroma_db") -> int:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = _extract_text(pdf_path)
    if not pages:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )
    all_texts = []
    all_metadatas = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append(page["metadata"])

    embeddings = OpenAIEmbeddings()
    Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=chroma_dir,
    )
    return len(all_texts)
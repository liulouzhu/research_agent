import os

import fitz
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_default_embeddings = None


def get_embeddings() -> OpenAIEmbeddings:
    global _default_embeddings
    if _default_embeddings is None:
        model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        emb_kwargs = {"model": model}
        emb_api_key = os.environ.get("EMBEDDING_API_KEY")
        emb_base_url = os.environ.get("EMBEDDING_BASE_URL")
        if emb_api_key:
            emb_kwargs["api_key"] = emb_api_key
        if emb_base_url:
            emb_kwargs["base_url"] = emb_base_url
        _default_embeddings = OpenAIEmbeddings(**emb_kwargs)
    return _default_embeddings


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

    embeddings = get_embeddings()
    Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=chroma_dir,
    )
    return len(all_texts)
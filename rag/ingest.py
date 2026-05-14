import os
import re

import fitz
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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


_SECTION_PATTERN = re.compile(
    r'^(\d+(?:\.\d+)*)[.\s]+(.+)$',
    re.MULTILINE | re.IGNORECASE,
)

_PLAIN_SECTION_KEYWORDS = {
    "abstract", "introduction", "conclusion", "references",
    "acknowledgements", "acknowledgment", "background", "related work",
    "method", "methods", "methodology", "results", "discussion",
    "experiment", "experiments", "evaluation", "appendix", "supplementary",
    "summary", "future work",
}


def _text_to_markdown(text: str) -> str:
    lines = text.split("\n")
    md_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            md_lines.append("")
            continue
        first_word = stripped.split()[0].rstrip(".").rstrip(",").rstrip(":").lower() if stripped else ""
        if first_word in _PLAIN_SECTION_KEYWORDS:
            md_lines.append(f"# {stripped}")
            continue
        m = _SECTION_PATTERN.match(stripped)
        if m:
            number = m.group(1)
            num_parts = number.split(".")
            depth = len(num_parts)
            depth = min(depth + 1, 3)
            md_lines.append(f"{'#' * depth} {stripped}")
        else:
            md_lines.append(stripped)
    return "\n".join(md_lines)


_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def ingest_document(pdf_path: str, chroma_dir: str = "./chroma_db") -> int:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = _extract_pages(pdf_path)
    if not pages:
        return 0

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        return_each_line=False,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )

    all_texts = []
    all_metadatas = []

    for page in pages:
        md_text = _text_to_markdown(page["text"])
        try:
            md_chunks = md_splitter.split_text(md_text)
        except Exception:
            md_chunks = []

        if not md_chunks:
            chunks = char_splitter.split_text(page["text"])
            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append(page["metadata"])
        else:
            for md_chunk in md_chunks:
                content = md_chunk.page_content
                header_meta = md_chunk.metadata
                combined_meta = {**page["metadata"], **header_meta}
                if len(content) > 1024:
                    sub_chunks = char_splitter.split_text(content)
                    for sub in sub_chunks:
                        all_texts.append(sub)
                        all_metadatas.append(combined_meta)
                else:
                    all_texts.append(content)
                    all_metadatas.append(combined_meta)

    if not all_texts:
        return 0

    embeddings = get_embeddings()
    Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=chroma_dir,
    )
    return len(all_texts)
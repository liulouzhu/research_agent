import os

import arxiv

from rag.ingest import ingest_document


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "authors": ", ".join(str(a.name) if hasattr(a, 'name') else str(a) for a in paper.authors[:3]),
            "summary": paper.summary[:300],
            "pdf_url": paper.pdf_url,
            "entry_id": paper.entry_id,
            "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
        })
    return results


def download_paper(entry_id: str, download_dir: str = "./downloads") -> str:
    os.makedirs(download_dir, exist_ok=True)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[entry_id])
    paper = next(client.results(search))
    filepath = paper.download_pdf(dirpath=download_dir)
    return filepath


def arxiv_agent(state: dict, llm, chroma_dir: str = "./chroma_db") -> dict:
    query = state["query"]
    prefs = state.get("user_preferences", {})
    lang = prefs.get("language", "chinese")

    search_results = search_arxiv(query)

    if not search_results:
        return {"output": "未在 arXiv 上找到相关论文。" if lang == "chinese" else "No papers found on arXiv."}

    result_lines = []
    for i, paper in enumerate(search_results, 1):
        result_lines.append(
            f"{i}. {paper['title']}\n"
            f"   作者: {paper['authors']}\n"
            f"   摘要: {paper['summary']}\n"
            f"   链接: {paper['pdf_url']}\n"
            f"   ID: {paper['entry_id']}"
        )
    search_output = "\n\n".join(result_lines)

    paper_ids = [r["entry_id"].split("/")[-1] for r in search_results]
    ids_str = ", ".join(paper_ids)

    if lang == "chinese":
        output = f"在 arXiv 上搜索到 {len(search_results)} 篇相关论文：\n\n{search_output}\n\n输入 `download <论文ID>` 可下载论文并自动入库。\n论文ID列表: {ids_str}"
    else:
        output = f"Found {len(search_results)} papers on arXiv:\n\n{search_output}\n\nType `download <paper_id>` to download and ingest.\nPaper IDs: {ids_str}"

    return {"output": output}
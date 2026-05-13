from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def retrieve(query: str, chroma_dir: str = "./chroma_db", top_k: int = 5) -> list[dict]:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    results = []
    for doc, score in docs_with_scores:
        results.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", -1),
                "score": score,
            }
        )
    return results
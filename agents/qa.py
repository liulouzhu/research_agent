from rag.retriever import retrieve


def qa_agent(state: dict, llm, chroma_dir: str = "./chroma_db") -> dict:
    query = state["query"]
    results = retrieve(query, chroma_dir)

    if not results:
        from langchain_core.messages import HumanMessage, SystemMessage
        msgs = [
            SystemMessage(content="你是一个科研助手。用户的问题在知识库中没有找到相关文档，请如实告知用户。"),
            HumanMessage(content=query),
        ]
        response = llm.invoke(msgs)
        return {"output": response.content, "context": []}

    context_text = "\n\n".join(
        f"[来源：{r['source']} 第{r['page']}页]\n{r['content']}" for r in results
    )

    from langchain_core.messages import HumanMessage, SystemMessage
    prefs = state.get("user_preferences", {})
    pref_str = f"用户偏好：语言={prefs.get('language', 'chinese')}，风格={prefs.get('style', 'academic')}"

    system_msg = SystemMessage(
        content=f"你是一个科研助手。根据以下检索到的文档片段回答用户问题。在回答中标注引用来源。{pref_str}\n\n检索结果：\n{context_text}"
    )
    user_msg = HumanMessage(content=query)

    history = state.get("messages", [])
    response = llm.invoke([system_msg] + [m for m in history if not isinstance(m, SystemMessage)] + [user_msg])

    return {
        "output": response.content,
        "context": [r["content"] for r in results],
    }
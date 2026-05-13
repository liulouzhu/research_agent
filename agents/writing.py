from langchain_core.messages import HumanMessage, SystemMessage


def writing_agent(state: dict, llm) -> dict:
    query = state["query"]
    prefs = state.get("user_preferences", {})
    pref_str = f"用户偏好：语言={prefs.get('language', 'chinese')}，风格={prefs.get('style', 'academic')}"

    system_msg = SystemMessage(
        content=f"你是一个学术论文写作助手。根据用户要求撰写论文内容，包括段落、大纲、草稿等。{pref_str}"
    )
    user_msg = HumanMessage(content=query)

    history = state.get("messages", [])
    messages = [system_msg] + [m for m in history if not isinstance(m, SystemMessage)] + [user_msg]
    response = llm.invoke(messages)

    return {"output": response.content}

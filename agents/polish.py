from langchain_core.messages import HumanMessage, SystemMessage


def polish_agent(state: dict, llm) -> dict:
    query = state["query"]
    prefs = state.get("user_preferences", {})
    pref_str = f"用户偏好：语言={prefs.get('language', 'chinese')}，风格={prefs.get('style', 'academic')}"

    system_msg = SystemMessage(
        content=f"你是一个学术论文润色助手。用户会提供需要润色/修改/优化的文本，请在保持原意的基础上改进表达。{pref_str}"
    )
    user_msg = HumanMessage(content=query)

    history = state.get("messages", [])
    messages = [system_msg] + [m for m in history if not isinstance(m, SystemMessage)] + [user_msg]
    response = llm.invoke(messages)

    return {"output": response.content}

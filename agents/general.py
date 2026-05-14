from langchain_core.messages import HumanMessage, SystemMessage


def general_agent(state: dict, llm) -> dict:
    query = state["query"]
    prefs = state.get("user_preferences", {})
    pref_str = f"用户偏好：语言={prefs.get('language', 'chinese')}，风格={prefs.get('style', 'academic')}"

    system_msg = SystemMessage(
        content=f"你是一个通用科研助手。帮助用户解答各类科研相关问题，包括概念解释、方法建议、工具推荐等。{pref_str}"
    )
    user_msg = HumanMessage(content=query)

    history = state.get("messages", [])
    messages = [system_msg] + [m for m in history if not isinstance(m, SystemMessage)] + [user_msg]
    response = llm.invoke(messages)

    return {"output": response.content}
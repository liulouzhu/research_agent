import re

from langchain_core.prompts import ChatPromptTemplate

_CLASSIFY_PROMPT = """你是一个意图分类器。根据用户输入，将其分为以下三类之一：

- qa：用户在提问、查找、搜索、检索信息（如"什么是...""如何...""为什么..."）
- writing：用户要求写、生成、续写、起草论文内容（如"帮我写...""生成大纲..."）
- polish：用户要求润色、修改、优化、改写已有文本（如"润色这段...""修改这段..."）

只输出类别标签（qa/writing/polish），不要输出其他内容。

用户输入：{query}"""

_VALID_INTENTS = {"qa", "writing", "polish"}


def classify_intent(state: dict, llm) -> dict:
    prompt = ChatPromptTemplate.from_template(_CLASSIFY_PROMPT)
    messages = prompt.format_messages(query=state["query"])
    response = llm.invoke(messages)
    raw = response.content.strip().lower()
    match = re.search(r"\b(qa|writing|polish)\b", raw)
    intent = match.group(1) if match else "qa"
    return {"intent": intent}


def route_by_intent(state: dict) -> str:
    intent = state.get("intent", "qa")
    mapping = {
        "qa": "qa_agent",
        "writing": "writing_agent",
        "polish": "polish_agent",
    }
    return mapping.get(intent, "qa_agent")
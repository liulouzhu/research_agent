import re

from langchain_core.prompts import ChatPromptTemplate

_CLASSIFY_PROMPT = """你是一个意图分类器。根据用户输入，将其分为以下六类之一：

- qa：用户在提问、查找、搜索、检索信息（如"什么是...""如何...""为什么..."）
- writing：用户要求写、生成、续写、起草论文内容（如"帮我写...""生成大纲..."）
- polish：用户要求润色、修改、优化、改写已有文本（如"润色这段...""修改这段..."）
- arxiv：用户想搜索、查找、下载 arXiv 上的论文（如"帮我找论文""搜索arxiv""下载论文"）
- innovation：用户想生成创新点、发现研究空缺、寻找研究方向的创新思路（如"帮我找创新点""有什么创新方向""创新idea"）
- general：不属于以上五类的一般性对话、闲聊、方法建议、概念解释等

只输出类别标签（qa/writing/polish/arxiv/innovation/general），不要输出其他内容。

用户输入：{query}"""

_VALID_INTENTS = {"qa", "writing", "polish", "arxiv", "innovation", "general"}


def classify_intent(state: dict, llm) -> dict:
    prompt = ChatPromptTemplate.from_template(_CLASSIFY_PROMPT)
    messages = prompt.format_messages(query=state["query"])
    response = llm.invoke(messages)
    raw = response.content.strip().lower()
    match = re.search(r"\b(qa|writing|polish|arxiv|innovation|general)\b", raw)
    intent = match.group(1) if match else "general"
    return {"intent": intent}


def route_by_intent(state: dict) -> str:
    intent = state.get("intent", "general")
    mapping = {
        "qa": "qa_agent",
        "writing": "writing_agent",
        "polish": "polish_agent",
        "arxiv": "arxiv_agent",
        "innovation": "innovation_agent",
        "general": "general_agent",
    }
    return mapping.get(intent, "general_agent")
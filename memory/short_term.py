from langchain_core.messages import BaseMessage, SystemMessage


def build_message_history(
    messages: list[BaseMessage], system_prompt: str
) -> list[dict]:
    result = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        role = "user" if msg.type == "human" else "assistant"
        result.append({"role": role, "content": msg.content})
    return result
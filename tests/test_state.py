from state import State


def test_state_has_required_fields():
    state: State = {
        "query": "什么是注意力机制",
        "intent": "qa",
        "context": ["注意力机制是一种..."],
        "messages": [],
        "output": "",
        "user_preferences": {"language": "chinese", "style": "academic"},
    }
    assert state["query"] == "什么是注意力机制"
    assert state["intent"] == "qa"
    assert state["context"] == ["注意力机制是一种..."]
    assert state["output"] == ""
    assert state["user_preferences"]["language"] == "chinese"


def test_state_messages_add_reducer():
    from langchain_core.messages import HumanMessage

    state: State = {
        "query": "",
        "intent": "",
        "context": [],
        "messages": [HumanMessage(content="hello")],
        "output": "",
        "user_preferences": {},
    }
    new_msg = HumanMessage(content="world")
    updated = {
        "messages": state["messages"] + [new_msg],
    }
    assert len(updated["messages"]) == 2
    assert updated["messages"][1].content == "world"


from memory.short_term import build_message_history


def test_build_message_history_empty():
    result = build_message_history([], "system prompt")
    assert len(result) == 1
    assert result[0]["role"] == "system"


def test_build_message_history_with_messages():
    from langchain_core.messages import HumanMessage, AIMessage

    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    result = build_message_history(msgs, "you are a helper")
    assert len(result) == 3
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert result[2]["role"] == "assistant"
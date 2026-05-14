import sys

from dotenv import load_dotenv

load_dotenv()

from graph import build_graph
from memory.long_term import LongTermMemory
from rag.ingest import ingest_document

_CHROMA_DIR = "./chroma_db"
_PREFS_PATH = "./user_prefs.json"


def _handle_upload(path: str) -> None:
    try:
        n = ingest_document(path, _CHROMA_DIR)
        print(f"文档已摄入，生成 {n} 个文本块。")
    except FileNotFoundError:
        print(f"文件不存在：{path}")
    except Exception as e:
        print(f"摄入失败：{e}")


def _handle_set(memory: LongTermMemory, key: str, value: str) -> None:
    memory.update(key, value)
    print(f"已设置 {key} = {value}")


def _handle_prefs(memory: LongTermMemory) -> None:
    prefs = memory.get_all()
    for k, v in prefs.items():
        print(f"  {k} = {v}")


def main():
    memory = LongTermMemory(_PREFS_PATH)
    graph = build_graph(chroma_dir=_CHROMA_DIR)

    print("科研助手已启动。输入问题开始对话，输入 quit 退出。")
    print("命令：upload <path> | set <key> <value> | prefs | quit")

    messages = []
    prefs = memory.load()

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input == "quit":
            print("再见！")
            break

        parts = user_input.split(maxsplit=1)
        cmd = parts[0]

        if cmd == "upload" and len(parts) > 1:
            _handle_upload(parts[1])
            continue

        if cmd == "set" and len(parts) > 1:
            kv = parts[1].split(maxsplit=1)
            if len(kv) == 2:
                _handle_set(memory, kv[0], kv[1])
            else:
                print("用法：set <key> <value>")
            continue

        if cmd == "prefs":
            _handle_prefs(memory)
            continue

        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=user_input))

        result = graph.invoke({
            "query": user_input,
            "intent": "",
            "context": [],
            "messages": messages,
            "output": "",
            "user_preferences": prefs,
        })

        print(f"\n助手：{result['output']}")

        from langchain_core.messages import AIMessage
        messages.append(AIMessage(content=result["output"]))


if __name__ == "__main__":
    main()

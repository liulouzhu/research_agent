# Research Agent

基于 LangGraph 的科研助手 CLI 工具，支持意图分类路由、RAG 问答、论文写作、润色修改。

## 架构

```
CLI 输入
  ↓
意图分类节点 (LLM 零样本分类)
  ↓ conditional edge
  ├── qa_agent          → RAG 检索 + 问答
  ├── writing_agent     → 论文写作
  └── polish_agent      → 润色修改
  ↓
输出到 CLI
```

单一 LangGraph StateGraph，扁平结构，3 个 agent 节点通过共享 State 传递数据。

## 功能

- **QA 问答**：上传 PDF 文档后，基于内容检索回答问题，标注引用来源
- **论文写作**：根据用户要求生成论文段落、大纲、草稿
- **润色修改**：对已有文本进行学术润色，支持偏好调整
- **意图分类**：自动识别用户意图，路由到对应 agent
- **多级记忆**：短期对话上下文 + 长期用户偏好持久化

## 安装

```bash
pip install -r requirements.txt
```

## 配置

设置 OpenAI API Key：

```bash
export OPENAI_API_KEY="sk-..."
```

## 使用

启动：

```bash
python main.py
```

交互命令：

| 命令 | 说明 |
|------|------|
| `upload <path>` | 上传 PDF 文档到知识库 |
| `set <key> <value>` | 设置用户偏好（如 `set language english`） |
| `prefs` | 查看当前偏好设置 |
| 直接输入文本 | 进入意图分类 → agent 流程 |
| `quit` | 退出 |

可设置的偏好项：

- `language`：偏好语言（默认 `chinese`）
- `style`：写作风格（默认 `academic`）
- `domain`：研究领域

## 项目结构

```
research_agent/
├── main.py              # CLI 入口 + 主循环
├── graph.py             # LangGraph 图定义
├── state.py             # State TypedDict
├── agents/
│   ├── classifier.py    # 意图分类节点
│   ├── qa.py            # qa_agent
│   ├── writing.py       # writing_agent
│   └── polish.py        # polish_agent
├── rag/
│   ├── ingest.py        # 文档摄入（解析+分块+向量化）
│   └── retriever.py     # RAG 检索
├── memory/
│   ├── short_term.py    # 消息历史管理
│   └── long_term.py     # 用户偏好持久化
├── chroma_db/           # Chroma 向量数据库目录
├── user_prefs.json      # 用户偏好文件（自动生成）
├── requirements.txt
└── tests/               # 测试
```

## 测试

```bash
pytest tests/ -v
```

> `test_ingest_creates_collection` 需要 `OPENAI_API_KEY` 环境变量才能运行。

## 技术栈

- **LangGraph** — 图编排框架
- **LangChain (OpenAI/Chroma/Text-Splitters)** — LLM 集成与 RAG
- **ChromaDB** — 向量数据库
- **PyMuPDF** — PDF 解析
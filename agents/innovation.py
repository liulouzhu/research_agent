import json
import os

from agents.arxiv import search_arxiv

_CANDIDATE_PROMPT = """你是一个科研创新点生成器。给定一个研究主题，生成3-5个创新点候选。

## 研究主题
{topic}

## 已有文献上下文
{context}

## 要求
- 候选之间必须覆盖不同的创新角度，不要生成语义相似的想法
- 每个候选必须说明与现有工作的关键区别
- 优先考虑具体且可验证的想法，而非模糊的方向
- 每个想法应能在一个小团队6-12个月内验证

## 输出格式
生成3-5个创新点候选，每个包含：
- title: 标题（20字以内）
- problem: 解决的具体问题（1-2句）
- idea: 核心思路（2-3句）
- key_difference: 与现有工作的具体区别（1-2句）
- expected_value: 预期影响（1-2句）

请以JSON数组格式输出，每个元素包含上述字段。只输出JSON数组，不要输出其他内容。"""

_NOVELTY_PROMPT = """你是一个科研新颖性分析师。给定一个创新点候选和相关论文，分析其新颖性。

## 创新点
标题：{title}
问题：{problem}
思路：{idea}

## 相关论文
{papers}

## 分析要求
评估该创新点与现有工作的重叠程度和创新程度。

## 输出格式
以JSON格式输出：
{{
  "novelty_level": "HIGH/MEDIUM/LOW",
  "overlap_dimensions": ["列出重叠维度：problem/method/setting/experiment"],
  "similarities": ["与现有工作的相似之处"],
  "differences": ["与现有工作的区别"],
  "novelty_conclusion": "新颖性结论（1-2句）"
}}
只输出JSON，不要输出其他内容。"""

_REVIEW_PROMPT = """你是一个科研评审专家。给定多个创新点候选及其新颖性分析，评审并给出最终推荐。

## 创新点候选及分析
{candidates}

## 评审要求
对每个候选打分（1-10分），并给出最终推荐。

## 输出格式
以JSON格式输出：
{{
  "reviews": [
    {{
      "title": "候选标题",
      "novelty_score": 8,
      "feasibility_score": 7,
      "impact_score": 6,
      "overall_score": 7.0,
      "recommendation": "PROCEED/REVISE/DROP",
      "comment": "简要评语"
    }}
  ],
  "top_recommendations": ["推荐的最佳候选标题列表"]
}}
只输出JSON，不要输出其他内容。"""

_REVIEWER_PROMPT = """你是一个独立的外部评审员。给定执行者（主模型）对创新点的评审结果，提供你的独立批判性评估。

## 执行者的评审结果
{review_data}

## 评审要求
对每个候选，给出：
1. 你是否同意执行者的评分（同意/不同意）
2. 你的简要理由（1-2句）
3. 最终推荐：PROCEED / REVISE / DROP

以JSON格式输出：
{{
  "assessments": [
    {{
      "title": "候选标题",
      "agree": true/false,
      "reasoning": "理由",
      "recommendation": "PROCEED/REVISE/DROP"
    }}
  ],
  "summary": "总体评估总结"
}}
只输出JSON，不要输出其他内容。"""


def _parse_json_robust(text: str):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None


def _generate_candidates(llm, topic: str, context: str = "") -> list[dict]:
    prompt = _CANDIDATE_PROMPT.format(topic=topic, context=context or "暂无已有文献上下文。")
    from langchain_core.messages import HumanMessage, SystemMessage
    msgs = [
        SystemMessage(content="你是一个科研创新点生成专家。请严格按照指定JSON格式输出。"),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(msgs)
    candidates = _parse_json_robust(response.content)
    if isinstance(candidates, list):
        return candidates
    if isinstance(candidates, dict) and "candidates" in candidates:
        return candidates["candidates"]
    return []


def _analyze_novelty(llm, candidate: dict, papers: list[dict]) -> dict:
    papers_text = "\n".join(
        f"- {p['title']} ({p.get('published', 'N/A')}): {p.get('summary', '')[:200]}"
        for p in papers[:5]
    ) if papers else "未找到相关论文。"
    prompt = _NOVELTY_PROMPT.format(
        title=candidate.get("title", ""),
        problem=candidate.get("problem", ""),
        idea=candidate.get("idea", ""),
        papers=papers_text,
    )
    from langchain_core.messages import HumanMessage, SystemMessage
    msgs = [
        SystemMessage(content="你是一个科研新颖性分析专家。请严格按照指定JSON格式输出。"),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(msgs)
    result = _parse_json_robust(response.content)
    if result and isinstance(result, dict):
        return result
    return {"novelty_level": "UNKNOWN", "novelty_conclusion": "无法解析新颖性分析结果"}


def _review_candidates(llm, candidates_with_novelty: list[dict]) -> dict:
    candidates_text = ""
    for i, item in enumerate(candidates_with_novelty, 1):
        c = item["candidate"]
        n = item["novelty"]
        candidates_text += f"\n### 候选 {i}: {c.get('title', '')}\n"
        candidates_text += f"- 问题: {c.get('problem', '')}\n"
        candidates_text += f"- 思路: {c.get('idea', '')}\n"
        candidates_text += f"- 关键区别: {c.get('key_difference', '')}\n"
        candidates_text += f"- 新颖性: {n.get('novelty_level', 'UNKNOWN')} - {n.get('novelty_conclusion', '')}\n"
    prompt = _REVIEW_PROMPT.format(candidates=candidates_text)
    from langchain_core.messages import HumanMessage, SystemMessage
    msgs = [
        SystemMessage(content="你是一个科研评审专家。请严格按照指定JSON格式输出。"),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(msgs)
    result = _parse_json_robust(response.content)
    if result and isinstance(result, dict):
        return result
    return {"reviews": [], "top_recommendations": []}


def _external_review(reviewer_llm, review_result: dict) -> dict:
    if reviewer_llm is None:
        return None
    review_text = json.dumps(review_result, ensure_ascii=False, indent=2)
    prompt = _REVIEWER_PROMPT.format(review_data=review_text)
    from langchain_core.messages import HumanMessage, SystemMessage
    msgs = [
        SystemMessage(content="你是一个独立的外部科研评审员。请严格按照指定JSON格式输出。"),
        HumanMessage(content=prompt),
    ]
    response = reviewer_llm.invoke(msgs)
    result = _parse_json_robust(response.content)
    if result and isinstance(result, dict):
        return result
    return None


def innovation_agent(state: dict, llm, reviewer_llm=None, chroma_dir: str = "./chroma_db") -> dict:
    topic = state["query"]
    prefs = state.get("user_preferences", {})
    lang = prefs.get("language", "chinese")

    context = ""
    arxiv_results = search_arxiv(topic, max_results=5)
    if arxiv_results:
        context_lines = []
        for p in arxiv_results:
            context_lines.append(f"- {p['title']} ({p.get('published', '')}): {p.get('summary', '')[:200]}")
        context = "\n".join(context_lines)

    candidates = _generate_candidates(llm, topic, context)
    if not candidates:
        no_result = "No innovation candidates generated." if lang != "chinese" else "未能生成创新点候选，请重试。"
        return {"output": no_result}

    candidates_with_novelty = []
    for c in candidates:
        novelty = _analyze_novelty(llm, c, arxiv_results)
        candidates_with_novelty.append({"candidate": c, "novelty": novelty})

    review_result = _review_candidates(llm, candidates_with_novelty)

    external_result = _external_review(reviewer_llm, review_result)

    output_parts = []
    if lang == "chinese":
        output_parts.append(f"# 研究主题：{topic}\n")
        output_parts.append("## 创新点候选\n")
        for i, item in enumerate(candidates_with_novelty, 1):
            c = item["candidate"]
            n = item["novelty"]
            output_parts.append(f"### {i}. {c.get('title', '')}")
            output_parts.append(f"- **问题**: {c.get('problem', '')}")
            output_parts.append(f"- **思路**: {c.get('idea', '')}")
            output_parts.append(f"- **关键区别**: {c.get('key_difference', '')}")
            output_parts.append(f"- **新颖性**: {n.get('novelty_level', 'UNKNOWN')} - {n.get('novelty_conclusion', '')}")
            output_parts.append("")
        if review_result.get("reviews"):
            output_parts.append("## 评审结果\n")
            for r in review_result["reviews"]:
                output_parts.append(f"- **{r.get('title', '')}** — 总分: {r.get('overall_score', 'N/A')} | 推荐: {r.get('recommendation', 'N/A')}")
                output_parts.append(f"  评语: {r.get('comment', '')}")
            if review_result.get("top_recommendations"):
                output_parts.append(f"\n**最佳推荐**: {', '.join(review_result['top_recommendations'])}")
        if external_result:
            output_parts.append("\n## 外部评审意见\n")
            for a in external_result.get("assessments", []):
                agree_str = "同意" if a.get("agree") else "不同意"
                output_parts.append(f"- **{a.get('title', '')}** — {agree_str}执行者评分 | 推荐: {a.get('recommendation', 'N/A')}")
                output_parts.append(f"  理由: {a.get('reasoning', '')}")
            if external_result.get("summary"):
                output_parts.append(f"\n总结: {external_result['summary']}")
    else:
        output_parts.append(f"# Research Topic: {topic}\n")
        output_parts.append("## Innovation Candidates\n")
        for i, item in enumerate(candidates_with_novelty, 1):
            c = item["candidate"]
            n = item["novelty"]
            output_parts.append(f"### {i}. {c.get('title', '')}")
            output_parts.append(f"- **Problem**: {c.get('problem', '')}")
            output_parts.append(f"- **Idea**: {c.get('idea', '')}")
            output_parts.append(f"- **Key Difference**: {c.get('key_difference', '')}")
            output_parts.append(f"- **Novelty**: {n.get('novelty_level', 'UNKNOWN')} - {n.get('novelty_conclusion', '')}")
            output_parts.append("")
        if review_result.get("reviews"):
            output_parts.append("## Review Results\n")
            for r in review_result["reviews"]:
                output_parts.append(f"- **{r.get('title', '')}** — Score: {r.get('overall_score', 'N/A')} | Rec: {r.get('recommendation', 'N/A')}")
                output_parts.append(f"  Comment: {r.get('comment', '')}")
            if review_result.get("top_recommendations"):
                output_parts.append(f"\n**Top Picks**: {', '.join(review_result['top_recommendations'])}")
        if external_result:
            output_parts.append("\n## External Review\n")
            for a in external_result.get("assessments", []):
                agree_str = "Agree" if a.get("agree") else "Disagree"
                output_parts.append(f"- **{a.get('title', '')}** — {agree_str} | Rec: {a.get('recommendation', 'N/A')}")
                output_parts.append(f"  Reasoning: {a.get('reasoning', '')}")
            if external_result.get("summary"):
                output_parts.append(f"\nSummary: {external_result['summary']}")

    return {"output": "\n".join(output_parts)}
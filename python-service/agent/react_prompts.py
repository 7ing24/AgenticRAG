"""ReAct Agent 提示词模板和格式化工具"""

import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from memory.memory_prompts import MEMORY_USAGE_PROMPT


class ReActPrompts:
    """ReAct Agent 的 prompt 模板和工具描述生成"""

    # ReAct 循环中需要 LLM 自主决策调用的工具
    REACT_TOOL_NAMES = {"knowledge_search", "question_rewrite", "long_term_memory_read"}

    SYSTEM_TEMPLATE = """You are an intelligent AI assistant with access to a knowledge base and tools.
Answer user questions by reasoning step by step and using the available tools to gather information.

## Available Tools
{tools}

## Current Date
{current_date}

## Tool Selection Guidelines
Choose tools strictly based on the question type:
- Objective facts / professional knowledge: Call `knowledge_search` (Optionally call `question_rewrite` first if the query is complex or ambiguous).
- Personal preference / user history / recall: Call `long_term_memory_read`. (Never use memory as a factual knowledge source).
- Mixed (needs both preference + knowledge): Call `knowledge_search` first, then `long_term_memory_read` if needed.
- Chit-chat / simple greetings: Answer directly with `Final Answer` without calling any tools.

## Response Format
For each step, you must output your thought followed by an action or final answer:

Thought: [Explain your reasoning]
Action: tool_name(param="value")

OR

Thought: [Explain your reasoning]
Final Answer: [Your complete response]

## Examples of Desired Behavior

Example 1 (Knowledge found):
Thought: The user is asking about Redis persistence. I need to search the knowledge base.
Action: knowledge_search(query="Redis RDB AOF persistence")
Observation: [Found 2 documents: [1] Redis supports RDB snapshots and AOF logs...]
Thought: The retrieved documents provide sufficient details about RDB and AOF. I can now answer.
Final Answer: Redis 主要支持两种持久化机制：1. RDB（快照机制）... 2. AOF（追加日志）...

Example 2 (Knowledge NOT found after search):
Thought: The user is asking about an internal project code. I will search the knowledge base.
Action: knowledge_search(query="Project Apollo architecture")
Observation: No documents found in knowledge base.
Thought: The first search returned no results. Let me try rewriting or searching a broader term.
Action: knowledge_search(query="Project Apollo")
Observation: No documents found in knowledge base.
Thought: The knowledge base does not contain any information about Project Apollo. I must inform the user.
Final Answer: 抱歉，知识库中暂无相关信息回答该问题。

## Important Rules
1. Always start your response with "Thought:" on a new line, followed by your reasoning.
2. Use EXACTLY ONE "Action:" line per response when using tools.
3. Information Sufficiency Standard:
   - If retrieved documents contain relevant concepts, mechanisms, or partial facts (e.g., mentioning JVM, bytecode, features for Java), synthesize a factual answer based on those contents.
   - Do NOT reject documents just because they don't provide a word-by-word standard dictionary definition.
4. Retry and Fallback:
   - If the first search returns 0 documents or completely irrelevant topics, you may retry ONCE with a broader keyword (e.g., search "Java" instead of "什么是Java").
   - Only when search results have NO relation to the topic whatsoever after retry, output strictly:
     Final Answer: 抱歉，知识库中暂无相关信息回答该问题。
5. Output Language: Respond in the same language as the user's question."""

    USER_QUESTION_TEMPLATE = """## Conversation Context
    {conversation_history}

    ## Current Question
    {question}

    ## Additional Context
    {context}"""

    @staticmethod
    def format_tools_for_llm(tool_registry, tool_filter: Optional[Set[str]] = None) -> str:
        """从 ToolRegistry 动态生成工具描述文本"""
        all_tools = tool_registry.get_all_tools()
        if tool_filter is not None:
            tools = {k: v for k, v in all_tools.items() if k in tool_filter}
        else:
            tools = all_tools

        if not tools:
            return "(No tools available)"

        lines = []
        for name, tool in tools.items():
            schema = tool_registry.get_tool_info(name)
            if not schema:
                continue

            params = []
            for param_name, prop in schema["input_schema"]["properties"].items():
                param_type = prop.get("type", "string")
                desc = prop.get("description", "")
                required = prop.get("required", False)
                default = prop.get("default")

                if default is not None:
                    param_str = f'{param_name}="{default}"' if isinstance(default, str) else f"{param_name}={default}"
                else:
                    param_str = param_name

                required_mark = " (required)" if required else ""
                params.append(f"  {param_str}: {param_type} - {desc}{required_mark}")

            params_block = "\n".join(params) if params else "  (no parameters)"
            lines.append(f"- **{name}**: {schema['description']}\n{params_block}")

        return "\n\n".join(lines)

    @staticmethod
    def format_observation(tool_name: str, result: Dict[str, Any], max_chars: int = 1500) -> str:
        """将工具返回结果格式化为 LLM 可读的 observation 文本"""
        if not result:
            return f"Tool '{tool_name}' returned empty result."

        try:
            if tool_name == "knowledge_search":
                return ReActPrompts._format_search_result(result, max_chars)

            elif tool_name == "question_rewrite":
                rewritten = result.get("rewritten_question", result.get("result", str(result)))
                return f"Rewritten query: {rewritten}"

            elif tool_name == "rerank":
                docs = result.get("reranked_documents", result.get("documents", []))
                return ReActPrompts._format_doc_list(docs, "Reranked", max_chars)

            elif tool_name in ("working_memory_read", "memory_read"):
                messages = result.get("messages", [])
                if messages:
                    lines = ["Conversation history:"]
                    for msg in messages[-10:]:
                        role = msg.get("role", "unknown")
                        content = str(msg.get("content", ""))[:200]
                        lines.append(f"  {role}: {content}")
                    return "\n".join(lines)[:max_chars]
                return "No conversation history found."

            elif tool_name == "working_memory_write":
                return f"Memory saved successfully: {result.get('status', 'ok')}"

            elif tool_name == "doc_summary":
                summary = result.get("summary", result.get("result", str(result)))
                return f"Document Summary: {summary}"[:max_chars]

            elif tool_name == "ocr_extract":
                text = result.get("text", result.get("result", str(result)))
                return f"Extracted Text: {text}"[:max_chars]

            elif tool_name == "long_term_memory_read":
                memories = (result or {}).get("memories") or {}
                if not any(memories.values()):
                    return "未检索到相关长期记忆，可直接基于当前对话回答。"
                return MEMORY_USAGE_PROMPT.format(
                    memories_dict=memories,
                    current_timestamp=int(time.time()),
                )[:max_chars]

            else:
                return str(result)[:max_chars]

        except Exception as e:
            return f"Error formatting observation: {e}\nRaw result: {str(result)[:500]}"

    @staticmethod
    def _format_search_result(result: Dict[str, Any], max_chars: int) -> str:
        """格式化知识检索结果"""
        docs = result.get("documents", result.get("chunks", result.get("results", [])))
        if not docs:
            return "No documents found in knowledge base."

        lines = [f"Found {len(docs)} documents:"]
        total_chars = 0

        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                content = doc.get("content", doc.get("page_content", str(doc)))
                metadata = doc.get("metadata", {})
                score = doc.get("score", metadata.get("score", "N/A"))
            else:
                content = getattr(doc, "page_content", str(doc))
                metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
                score = metadata.get("score", "N/A")

            source = metadata.get("source", metadata.get("doc_name", "Unknown"))
            page = metadata.get("page", metadata.get("page_number", ""))

            content_preview = str(content)[:300].replace("\n", " ")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)

            line = f"\n[{i + 1}] (score: {score_str}) {content_preview}"
            if source:
                line += f"\n    Source: {source}"
                if page:
                    line += f", page {page}"

            total_chars += len(line)
            if total_chars > max_chars:
                lines.append("\n... (truncated)")
                break

            lines.append(line)

        return "".join(lines)

    @staticmethod
    def _format_doc_list(docs: list, label: str, max_chars: int) -> str:
        """通用文档列表格式化"""
        if not docs:
            return f"{label}: No documents."

        lines = [f"{label} {len(docs)} documents:"]
        total_chars = 0
        for i, doc in enumerate(docs[:10]):
            if isinstance(doc, dict):
                content = str(doc.get("content", doc.get("page_content", str(doc))))[:200]
                score = doc.get("score", "N/A")
            else:
                content = str(getattr(doc, "page_content", str(doc)))[:200]
                score = getattr(doc, "score", "N/A")

            line = f"\n[{i + 1}] {content.replace(chr(10), ' ')} (score: {score})"
            total_chars += len(line)
            if total_chars > max_chars:
                lines.append("\n... (truncated)")
                break
            lines.append(line)

        return "".join(lines)

    @staticmethod
    def build_initial_messages(
        question: str,
        context: str,
        conversation_history: str,
        tool_registry,
        user_profile: str = "",
    ) -> List[Dict[str, str]]:
        """构建 ReAct 循环的初始消息列表：[system, user]"""
        tools_desc = ReActPrompts.format_tools_for_llm(
            tool_registry, tool_filter=ReActPrompts.REACT_TOOL_NAMES
        )

        system_prompt = ReActPrompts.SYSTEM_TEMPLATE.format(
            tools=tools_desc,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )
        if user_profile:
            system_prompt += (
                f"\n\n## User Profile (Background Context Only)\n"
                f"{user_profile}\n"
                f"CRITICAL: When the knowledge base has no relevant information, output ONLY '抱歉，知识库中暂无相关信息回答该问题。' and STOP immediately. "
                f"Do NOT use user profile to extrapolate, explain missing criteria, or expand on system architecture details."
            )

        user_message = ReActPrompts.USER_QUESTION_TEMPLATE.format(
            question=question,
            context=context or "(none)",
            conversation_history=conversation_history or "(new conversation)",
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
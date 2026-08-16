"""ReAct Agent 提示词模板和格式化工具"""

from typing import Dict, Any
from datetime import datetime


class ReActPrompts:
    """ReAct Agent 的 prompt 模板和工具描述生成"""

    # ReAct 循环中实际需要 LLM 自主决策调用的工具
    # 记忆读写由 MemoryAgent 在循环外预加载/后写入，rerank 已内嵌在 knowledge_search 中
    REACT_TOOL_NAMES = {"knowledge_search", "question_rewrite"}

    SYSTEM_TEMPLATE = """You are an intelligent AI assistant with access to a knowledge base and tools.
Answer user questions by reasoning step by step and using the available tools to gather information.

## Available Tools
{tools}

## Current Date
{current_date}

## Response Format
You MUST respond in EXACTLY the following format:

When you need to use a tool:
Thought: <your step-by-step reasoning about what you need to know>
Action: tool_name(param1="value1", param2=5)

When you have enough information to answer:
Thought: <brief concluding reasoning>
Final Answer: <your complete answer to the user>

## Important Rules
1. Always start with "Thought:" on a new line
2. Use EXACTLY one "Action:" line per response when using tools. Parameters use key="value" for strings, key=number for numbers
3. After each Action, you will receive an "Observation:" with the tool result
4. Use "Final Answer:" ONLY when you have gathered enough information
5. NEVER invent information. Base your answer on observations or your own knowledge only when no relevant documents exist
6. Respond in the same language as the user's question
7. Include citations by referencing document names or IDs from observations when available
8. For greetings or simple chat, you may answer directly with Final Answer without calling tools
9. If tool results are insufficient, try a different query or approach rather than giving up
10. Question_rewrite only reformulates your query — it does NOT return any document content. You MUST call knowledge_search afterwards to retrieve actual information. Never output Final Answer immediately after question_rewrite."""

    USER_QUESTION_TEMPLATE = """## Conversation Context
{conversation_history}

## Current Question
{question}

## Additional Context
{context}

Begin reasoning step by step. Remember the format:
Thought: <reasoning>
Action: tool_name(param="value")
or
Final Answer: <answer>"""

    @staticmethod
    def format_tools_for_llm(tool_registry, tool_filter: set = None) -> str:
        """从 ToolRegistry 动态生成工具描述文本

        Args:
            tool_registry: 工具注册中心
            tool_filter: 可选，只展示指定名称的工具；不传则展示全部已注册工具
        """
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

            # 构建参数列表
            params = []
            for param_name, prop in schema["input_schema"]["properties"].items():
                param_type = prop.get("type", "string")
                desc = prop.get("description", "")
                required = prop.get("required", False)
                default = prop.get("default")

                if default is not None:
                    if isinstance(default, str):
                        param_str = f'{param_name}="{default}"'
                    else:
                        param_str = f'{param_name}={default}'
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
                    for msg in messages[-10:]:  # last 10 messages
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
            doc_id = metadata.get("doc_id", "")
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
    ) -> str:
        """构建 ReAct 循环的初始对话文本"""
        tools_desc = ReActPrompts.format_tools_for_llm(
            tool_registry, tool_filter=ReActPrompts.REACT_TOOL_NAMES
        )

        system_prompt = ReActPrompts.SYSTEM_TEMPLATE.format(
            tools=tools_desc,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

        user_message = ReActPrompts.USER_QUESTION_TEMPLATE.format(
            question=question,
            context=context or "(none)",
            conversation_history=conversation_history or "(new conversation)",
        )

        return system_prompt + "\n\n" + user_message

from tools.registry import tool_registry
from tools.base import Tool, ToolSchema, SchemaProperty, ToolMetadata
from tools.question_rewrite import QuestionRewriteTool
from tools.knowledge_search import KnowledgeSearchTool
from tools.rerank import RerankTool
from tools.memory_read import WorkingMemoryReadTool
from tools.memory_read import LongTermMemoryReadTool
from tools.memory_write import WorkingMemoryWriteTool
from tools.doc_summary import DocSummaryTool
from tools.ocr_extract import OCRExtractTool as OCRTool
from tools.execution import tool_execution_tracker


def register_all_tools():
    """注册所有可用工具"""
    tools = [
        QuestionRewriteTool(),
        KnowledgeSearchTool(),
        RerankTool(),
        WorkingMemoryReadTool(),
        LongTermMemoryReadTool(),
        WorkingMemoryWriteTool(),
        DocSummaryTool(),
        OCRTool(),
    ]

    for tool in tools:
        try:
            tool_registry.register_tool(tool)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to register tool {tool.name}: {e}")


register_all_tools()


__all__ = [
    "tool_registry",
    "tool_execution_tracker",
    "Tool",
    "ToolSchema",
    "SchemaProperty",
    "ToolMetadata",
    "QuestionRewriteTool",
    "KnowledgeSearchTool",
    "RerankTool",
    "WorkingMemoryReadTool",
    "LongTermMemoryReadTool",
    "WorkingMemoryWriteTool",
    "DocSummaryTool",
    "OCRTool",
    "register_all_tools",
]

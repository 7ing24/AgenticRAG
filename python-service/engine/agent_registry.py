"""Agent 注册中心 — 注册、查找、按能力匹配 Agent"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Agent 注册中心（单例）

    所有 worker agent 在此注册，MultiAgentOrchestrator 通过 match() 找到最合适的 worker。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}  # name → {"agent": ..., "capabilities": [...], "description": str}
        return cls._instance

    def register(self, name: str, agent, capabilities: List[str], description: str = ""):
        """注册 agent

        Args:
            name: agent 名称
            agent: agent 实例
            capabilities: 能力标签列表，如 ["knowledge_search", "reasoning", "multi_step"]
            description: agent 能力描述，用于 LLM 匹配
        """
        self._agents[name] = {
            "agent": agent,
            "capabilities": capabilities,
            "description": description or name,
        }
        logger.info(f"[AgentRegistry] Registered agent '{name}' with capabilities: {capabilities}")

    def get(self, name: str):
        """按名称获取 agent"""
        entry = self._agents.get(name)
        return entry["agent"] if entry else None

    def match(self, sub_task: str) -> Any:
        """根据 sub-task 描述匹配最合适的 agent

        优先 LLM 匹配，失败时返回默认的 ReActAgent。
        """
        if not self._agents:
            logger.warning("[AgentRegistry] No agents registered")
            return None

        # 尝试 LLM 匹配
        agent = self._match_with_llm(sub_task)
        if agent is not None:
            return agent

        # fallback：返回第一个注册的 agent（通常是 ReActAgent）
        fallback_name = list(self._agents.keys())[0]
        logger.info(f"[AgentRegistry] Match fallback to '{fallback_name}' for: {sub_task[:50]}...")
        return self._agents[fallback_name]["agent"]

    def match_all(self, sub_tasks: List[str]) -> Dict[str, Any]:
        """为多个 sub-task 批量匹配 agent

        Returns:
            {sub_task: agent_instance}
        """
        return {task: self.match(task) for task in sub_tasks}

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有注册的 agent"""
        return [
            {
                "name": name,
                "capabilities": info["capabilities"],
                "description": info["description"],
            }
            for name, info in self._agents.items()
        ]

    def _match_with_llm(self, sub_task: str):
        """使用 LLM 根据能力描述匹配最合适的 agent"""
        if len(self._agents) == 0:
            return None
        if len(self._agents) == 1:
            return list(self._agents.values())[0]["agent"]

        # 构建 agent 能力描述
        agent_descriptions = "\n".join([
            f"- {name}: {info['description']} (capabilities: {', '.join(info['capabilities'])})"
            for name, info in self._agents.items()
        ])

        try:
            from core.llm import llm_service
            if not llm_service.llm:
                return None

            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = PromptTemplate.from_template(
                """根据子任务的需求，从以下 agent 中选择最合适的一个。

子任务：{sub_task}

可选 agent：
{agent_descriptions}

只回复 agent 名称，不要其他内容。"""
            )

            chain = prompt | llm_service.llm | StrOutputParser()
            agent_name = chain.invoke({
                "sub_task": sub_task,
                "agent_descriptions": agent_descriptions
            }).strip()

            # 验证返回的 agent 名称
            for name in self._agents:
                if name in agent_name:
                    logger.info(f"[AgentRegistry] LLM matched '{name}' for: {sub_task[:50]}...")
                    return self._agents[name]["agent"]

        except Exception as e:
            logger.warning(f"[AgentRegistry] LLM match failed: {e}")

        return None


# 模块级单例
agent_registry = AgentRegistry()

"""LLM 语义聚类 - 向量相似度粗聚类 + LLM 主题命名

对未命中问题做语义聚类：
1. 批量 embedding（复用 vector_store 的 embedding 模型）
2. 余弦相似度贪心聚类（纯计算，快、确定）
3. LLM 给每个簇生成主题名（只喂代表问题，token 可控）
"""

import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SemanticCluster:
    """未命中问题的语义聚类"""

    SIMILARITY_THRESHOLD = 0.75  # 余弦相似度阈值（0.75：簇更粗，减少同主题变体问法被拆散）
    TOP_REPRESENTATIVE = 5      # 每个簇取 count 最高的代表问题数

    # 需要从关键词中剔除的疑问词/停用词（作为前缀或整体出现）
    QUESTION_PREFIXES = [
        "什么是", "是什么", "如何", "怎么", "怎样", "怎么样",
        "为什么", "为何", "是否", "请问", "什么", "哪个", "哪些",
        "多少", "能不能", "可以", "有没有", "有无",
    ]
    STOPWORDS = {
        "什么", "如何", "怎么", "怎样", "怎么样", "为什么", "为何",
        "是否", "请问", "哪个", "哪些", "多少", "能不能", "可以",
        "有没有", "有无", "是", "的", "了", "吗", "呢", "啊",
    }

    def __init__(self):
        self._embeddings = None
        self._llm_service = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            from core.vector_store import vector_store_manager
            self._embeddings = vector_store_manager.embeddings
        return self._embeddings

    @property
    def llm_service(self):
        if self._llm_service is None:
            from core.llm import LLMService
            self._llm_service = LLMService()
        return self._llm_service

    def cluster(self, questions: List[Dict], threshold: Optional[float] = None) -> List[Dict]:
        """对未命中问题做语义聚类

        Args:
            questions: [{"question": str, "count": int}, ...]
            threshold: 余弦相似度阈值（0~1），默认用 SIMILARITY_THRESHOLD

        Returns:
            [{"topic": str, "questions": [str], "total_count": int}, ...]
            按 total_count 降序
        """
        if not questions:
            return []

        if threshold is None:
            threshold = self.SIMILARITY_THRESHOLD

        # 1. 批量 embedding
        texts = [q.get("question", "") for q in questions]
        try:
            import numpy as np
            vectors = np.array(self.embeddings.embed_documents(texts))
        except Exception as e:
            logger.warning(f"[SemanticCluster] Embedding failed, return empty: {e}")
            return []

        # 2. 余弦相似度贪心聚类
        groups = self._greedy_cluster(vectors, threshold)

        # 3. LLM 给每个簇命名 + 摘要 + 关键词
        clusters = []
        for group in groups:
            group_questions = [questions[i] for i in group]
            topic, summary, keywords = self._generate_topic(group_questions)
            clusters.append({
                "topic": topic,
                "summary": summary,
                "keywords": keywords,
                "questions": [q["question"] for q in group_questions],
                "total_count": sum(q.get("count", 0) for q in group_questions),
            })

        clusters.sort(key=lambda c: c["total_count"], reverse=True)
        return clusters

    def _greedy_cluster(self, vectors, threshold: float) -> List[List[int]]:
        """余弦相似度贪心聚类，返回每簇的下标列表"""
        import numpy as np

        n = len(vectors)
        # 归一化向量，方便算余弦相似度
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms

        processed = [False] * n
        groups = []

        for i in range(n):
            if processed[i]:
                continue
            group = [i]
            processed[i] = True
            for j in range(i + 1, n):
                if processed[j]:
                    continue
                similarity = float(np.dot(normalized[i], normalized[j]))
                if similarity >= threshold:
                    group.append(j)
                    processed[j] = True
            groups.append(group)

        return groups

    def _generate_topic(self, group_questions: List[Dict]) -> tuple:
        """LLM 给一个簇生成主题名、摘要、关键词，返回 (topic, summary, keywords)"""
        top = sorted(group_questions, key=lambda q: q.get("count", 0), reverse=True)[:self.TOP_REPRESENTATIVE]
        top_text = "\n".join(
            f"- {q['question']}（{q.get('count', 0)}次）" for q in top
        )

        prompt = f"""以下是几条用户提问但系统未回答的问题：

{top_text}

请为这些未命中问题做归纳，返回 JSON 格式：
{{"topic": "简洁主题名（2-8字）", "summary": "一句话主题摘要", "keywords": ["关键词1", "关键词2", "关键词3"]}}

要求：
- keywords 只保留实体名词/技术术语等核心词，如产品名、技术名、专有名词。
- 不要提取疑问词或泛化词，例如"什么是""如何""怎么""为什么""哪些""能否"等。

只返回 JSON，不要其他内容。"""

        try:
            result = self.llm_service.generate(prompt)
            if result and result.strip():
                data = json.loads(result.strip())
                topic = str(data.get("topic", "")).strip()
                summary = str(data.get("summary", "")).strip()
                keywords = data.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [keywords]
                if not topic:
                    raise ValueError("empty topic")
                cleaned = self._clean_keywords(keywords) if isinstance(keywords, list) else []
                return topic, summary, cleaned
        except Exception as e:
            logger.warning(f"[SemanticCluster] Topic generation failed: {e}")

        # fallback：用第一个问题截断
        first = group_questions[0].get("question", "未分类")
        topic = first[:8] if len(first) > 8 else first
        return topic, topic, []

    def _clean_keywords(self, keywords: List[str]) -> List[str]:
        """剔除关键词中的疑问词/停用词，去前缀、去重"""
        cleaned = []
        seen = set()
        for kw in keywords:
            kw = str(kw).strip()
            if not kw:
                continue
            # 去掉疑问词前缀（如"什么是RDB" -> "RDB"）
            for prefix in self.QUESTION_PREFIXES:
                if kw.startswith(prefix) and len(kw) > len(prefix):
                    kw = kw[len(prefix):].strip()
                    break
            # 清洗后仍为空或纯停用词则丢弃
            if not kw or kw in self.STOPWORDS:
                continue
            if kw not in seen:
                seen.add(kw)
                cleaned.append(kw)
        return cleaned


# 模块级单例
semantic_cluster = SemanticCluster()

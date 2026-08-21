"""数据集构建 — 从知识库分块自动生成评测 QA 对

关键设计:
  - QA 生成用 qwen-turbo（与被测的 qwen-plus 分离，避免自评偏差）
  - relevant_chunk_id 直接来自 chunk 元数据，无需事后映射
  - 每个 chunk 最多生成 1-2 个 QA 对

输出格式:
  {question, ground_truth, relevant_chunk_id, synthesizer_name}
"""

import json
import os
import re
import logging
import random
from typing import List, Dict, Any

from langchain_core.documents import Document as LCDocument
from langchain_community.llms import Tongyi

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

QA_GENERATION_PROMPT = """你是一个专业的QA测试集生成器。根据以下文本片段生成{n}个问题，用于评测RAG系统的检索和回答能力。

要求：
1. 问题是自然语言提问，类似真实用户会问的问题
2. 问题必须能用片段中的内容回答
3. 如果原文是代码，请生成关于代码功能、设计思路、算法原理的问题，而不是变量名、参数值
4. 一半是事实型(simple)：答案能直接从片段中找到
5. 一半是理解型(complex)：需要理解片段整体含义才能回答
6. 问题用中文

文本片段：
{context}

严格返回JSON数组（不要其他文字），每个对象包含question和difficulty两个字段：
[{{"question": "完整的中文问题", "difficulty": "simple"}}]"""


class EvalDatasetBuilder:
    """从知识库分块生成评测数据集"""

    def __init__(self):
        from core.config import config
        self._api_key = config.DASHSCOPE_API_KEY
        self._qa_llm_model = "qwen-turbo"

    def _get_qa_llm(self) -> Tongyi:
        return Tongyi(model_name=self._qa_llm_model, api_key=self._api_key, streaming=False)

    def build(self, n: int = 100, output_name: str = "synthetic_qa.json") -> List[Dict[str, Any]]:
        """一站式构建: 采样 chunk → LLM 生成 QA → 保存"""
        existing = self.load(output_name)
        if len(existing) >= n and all(d.get("relevant_chunk_id") and d.get("ground_truth") for d in existing):
            logger.info(f"Using cached dataset: {len(existing[:n])} samples")
            random.shuffle(existing)
            return existing[:n]

        chunks = self._sample_chunks(n)
        if not chunks:
            logger.error("No chunks found in knowledge base")
            return []

        logger.info(f"Sampled {len(chunks)} chunks, generating QA pairs with {self._qa_llm_model}...")

        dataset = self._generate_qa(chunks, n)

        if not dataset:
            logger.error("LLM generated 0 QA pairs")
            return []

        self._save(dataset, output_name)
        logger.info(f"Saved {len(dataset)} QA pairs to {output_name}")
        return dataset

    # ── QA 生成 ──────────────────────────────────────────────

    # 每个 chunk 最多生成的 QA 数量（避免单个 API 调用过多导致超时）
    MAX_QA_PER_CHUNK = 5

    def _generate_qa(self, chunks: List[LCDocument], n: int) -> List[Dict]:
        """用独立 LLM 从每个 chunk 生成 QA 对

        每个 chunk 最多生成 MAX_QA_PER_CHUNK 个 QA，需要的 chunks 不足时
        允许同一文档的多个 chunk 参与，确保多样性。
        """
        llm = self._get_qa_llm()
        target_per_chunk = min(self.MAX_QA_PER_CHUNK, max(1, n // max(len(chunks), 1)))

        dataset = []
        generated_count = 0
        for i, chunk in enumerate(chunks):
            context = chunk.page_content
            if len(context) > 4000:
                context = context[:4000]

            # 跳过纯代码 chunk（中文占比 < 2% 且行数 > 30）
            chinese_ratio = sum(1 for c in context if '\u4e00' <= c <= '\u9fff') / max(len(context), 1)
            lines = context.count('\n') + 1
            if chinese_ratio < 0.02 and lines > 30:
                logger.info(f"[{i + 1}/{len(chunks)}] Skipping code-dominant chunk")
                continue

            prompt = QA_GENERATION_PROMPT.format(n=target_per_chunk, context=context)

            try:
                response = llm.invoke(prompt)
                questions = self._parse_questions(response)
            except Exception as e:
                logger.warning(f"[{i + 1}/{len(chunks)}] LLM failed: {e}")
                continue

            doc_id = chunk.metadata.get("doc_id", "")
            parent_id = chunk.metadata.get("parent_id", "")
            if parent_id:
                relevant_chunk_id = parent_id
            else:
                relevant_chunk_id = f"doc_{doc_id}_chunk_{chunk.metadata.get('chunk_index', 0)}"
            # ground_truth 直接使用 chunk 原文，不让 LLM 生成
            ground_truth = context

            for q in questions:
                question_text = q.get("question", "").strip()
                if not question_text:
                    continue

                dataset.append({
                    "question": question_text,
                    "ground_truth": ground_truth,
                    "relevant_chunk_id": relevant_chunk_id,
                    "synthesizer_name": q.get("difficulty", "llm_generated"),
                })
                generated_count += 1

            logger.info(
                f"[{i + 1}/{len(chunks)}] {len(questions)} QA for {relevant_chunk_id}"
            )

            # 达到目标数量就停
            if generated_count >= n:
                break

        logger.info(f"Total: {len(dataset)} QA pairs from {len(chunks)} chunks")
        return dataset

    @staticmethod
    def _parse_questions(response: str) -> List[Dict]:
        """解析 LLM 返回的问题 JSON（只需 question 和 difficulty）"""
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            response = json_match.group(1)

        arr_match = re.search(r'\[[\s\S]*\]', response)
        if arr_match:
            response = arr_match.group(0)

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON. Response prefix: {response[:200]}...")
            return []

        if not isinstance(parsed, list):
            parsed = [parsed] if isinstance(parsed, dict) else []

        result = []
        for item in parsed:
            if isinstance(item, dict) and "question" in item:
                result.append({
                    "question": item["question"],
                    "difficulty": item.get("difficulty", "llm_generated"),
                })
        return result

    # ── 文档采样 ─────────────────────────────────────────────

    def _sample_chunks(self, n: int) -> List[LCDocument]:
        """从数据库采样 chunk

        优先每个 doc 取一个 chunk（保证多样性），doc 不够时同文档多 chunk 补齐。
        """
        try:
            from core.mysql_client import mysql_client

            rows = mysql_client.fetch_all(
                "SELECT doc_id, parent_id, chunk_index, chunk_text FROM knowledge_chunk ORDER BY chunk_index"
            )
            if not rows:
                return []

            by_doc: Dict[int, list] = {}
            for r in rows:
                by_doc.setdefault(r.get("doc_id"), []).append(r)

            doc_ids = list(by_doc.keys())
            random.shuffle(doc_ids)

            # 第一轮: 每个 doc 随机取一个 chunk
            chunks = []
            for doc_id in doc_ids:
                row = random.choice(by_doc[doc_id])
                if row.get("chunk_text"):
                    chunks.append(LCDocument(
                        page_content=row["chunk_text"],
                        metadata={
                            "doc_id": doc_id,
                            "chunk_index": row.get("chunk_index", 0),
                            "parent_id": row.get("parent_id", ""),
                        }
                    ))

            # 不够的话，从已有文档中再随机取不同 chunk
            while len(chunks) < n:
                doc_id = random.choice(doc_ids)
                candidates = [r for r in by_doc[doc_id]
                              if r.get("chunk_text")
                              and r.get("chunk_index") not in
                              {c.metadata["chunk_index"] for c in chunks
                               if c.metadata["doc_id"] == doc_id}]
                if not candidates:
                    # 该文档所有 chunk 都已采样，换一个
                    remaining = [did for did in doc_ids
                                 if any(r.get("chunk_index") not in
                                        {c.metadata["chunk_index"] for c in chunks
                                         if c.metadata["doc_id"] == did}
                                        for r in by_doc[did] if r.get("chunk_text"))]
                    if not remaining:
                        break  # 所有文档的所有 chunk 都已采样
                    doc_id = random.choice(remaining)
                    candidates = [r for r in by_doc[doc_id]
                                  if r.get("chunk_text")
                                  and r.get("chunk_index") not in
                                  {c.metadata["chunk_index"] for c in chunks
                                   if c.metadata["doc_id"] == doc_id}]
                    if not candidates:
                        break
                row = random.choice(candidates)
                chunks.append(LCDocument(
                    page_content=row["chunk_text"],
                    metadata={
                        "doc_id": doc_id,
                        "chunk_index": row.get("chunk_index", 0),
                    }
                ))

            logger.info(f"Sampled {len(chunks)} chunks from {len(doc_ids)} documents")
            return chunks
        except Exception as e:
            logger.warning(f"MySQL chunk sampling failed: {e}")
            return self._sample_from_vector_store(n)

    def _sample_from_vector_store(self, n: int) -> List[LCDocument]:
        """降级: 从 Milvus 采样（通过 MySQL 获取更可靠）"""
        return []
        except Exception as e:
            logger.error(f"Vector store sampling failed: {e}")
            return []

    # ── 持久化 ───────────────────────────────────────────────

    @staticmethod
    def load(filename: str = "synthetic_qa.json") -> List[Dict[str, Any]]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    @staticmethod
    def _save(dataset: List[Dict], filename: str):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, filename), "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


def build_eval_dataset(n: int = 50) -> List[Dict[str, Any]]:
    """一站式构建评测数据集"""
    builder = EvalDatasetBuilder()
    dataset = builder.load("synthetic_qa.json")
    if len(dataset) >= n and all(d.get("relevant_chunk_id") and d.get("ground_truth") for d in dataset):
        logger.info(f"Using cached dataset: {len(dataset[:n])} samples")
        random.shuffle(dataset)
        return dataset[:n]
    return builder.build(n=n)

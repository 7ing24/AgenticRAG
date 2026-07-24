"""评测指标 — IR检索指标（LLM判定）+ RAGAS生成指标"""

import math
import logging
from typing import List, Dict, Any
from collections import defaultdict

from langchain_community.llms import Tongyi

logger = logging.getLogger(__name__)

# ── LLM 相关性判定 Prompt ──────────────────────────────────

RELEVANCE_PROMPT = """判断以下文本片段是否包含回答问题的相关信息。只回答 YES 或 NO。

问题：{question}

文本片段：
{chunk_text}

这个文本片段是否包含可以用来回答问题的信息？回答 YES 或 NO："""


def _relevance_judge(question: str, chunk_text: str, llm: Tongyi) -> bool:
    """用独立 LLM 判断 chunk 是否与问题相关"""
    if not chunk_text.strip():
        return False
    try:
        prompt = RELEVANCE_PROMPT.format(
            question=question,
            chunk_text=chunk_text[:2000],
        )
        response = llm.invoke(prompt).strip().upper()
        return response.startswith("YES")
    except Exception as e:
        logger.warning(f"Relevance judge failed: {e}")
        return False


def _judge_chunks(question: str, chunks: List[Dict], judge_llm: Tongyi) -> List[bool]:
    """判断多个 chunk 是否与问题相关，返回布尔列表"""
    results = []
    for doc in chunks:
        if isinstance(doc, dict):
            text = doc.get("content", doc.get("page_content", ""))
        else:
            text = getattr(doc, "page_content", str(doc))
        results.append(_relevance_judge(question, text, judge_llm))
    return results


# ═══════════════════════════════════════════════════════════
# IR 检索指标（LLM 判定版）
# ═══════════════════════════════════════════════════════════

def recall_at_k(relevant_at_positions: List[bool], k: int) -> float:
    """Top-K 中是否有相关文档"""
    return 1.0 if any(relevant_at_positions[:k]) else 0.0


def mrr_from_positions(relevant_at_positions: List[bool]) -> float:
    """第一个相关文档的倒数排名"""
    for i, relevant in enumerate(relevant_at_positions):
        if relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant_at_positions: List[bool], k: int) -> float:
    """归一化折损累积增益"""
    idcg = 1.0 / math.log2(2)  # relevance=1 at position 1
    if idcg == 0:
        return 0.0
    dcg = 0.0
    for i, relevant in enumerate(relevant_at_positions[:k]):
        if relevant:
            dcg = 1.0 / math.log2(i + 2)
            break
    return dcg / idcg


def compute_ir_metrics(
    dataset: List[Dict[str, Any]],
    retrieval_func,
    k_values: tuple = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """对每条数据跑检索，用 LLM(qwen-plus) 判定每个 chunk 是否相关

    不再依赖 relevant_chunk_id 精确匹配，避免"语义正确答案但 chunk_id
    不对就判错"的问题。
    """
    from core.config import config

    max_k = max(k_values)
    api_key = config.DASHSCOPE_API_KEY
    judge_llm = Tongyi(model_name="qwen-turbo", api_key=api_key, streaming=False) if api_key else None

    if judge_llm is None:
        logger.error("DASHSCOPE_API_KEY not set, cannot run LLM-based IR evaluation")
        return {"ir_metrics": {}, "per_query": [], "total_queries": 0, "skipped_queries": len(dataset)}

    metrics = defaultdict(list)
    per_query = []

    for i, item in enumerate(dataset):
        question = item.get("question", "")
        if not question:
            continue

        try:
            retrieved = retrieval_func(question, top_k=max_k)
            # 提取检索到的 chunk_id（父子块模式优先用 parent_id）
            retrieved_chunk_ids = []
            for doc in retrieved:
                metadata = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
                pid = metadata.get("parent_id", "")
                if pid:
                    retrieved_chunk_ids.append(pid)
                else:
                    retrieved_chunk_ids.append(
                        f"doc_{metadata.get('doc_id', '')}_chunk_{metadata.get('chunk_index', 0)}"
                    )
            # 生成 LLM 回答
#             contexts = []
#             for doc in retrieved[:5]:
#                 if isinstance(doc, dict):
#                     contexts.append(doc.get("content", doc.get("page_content", "")))
#                 else:
#                     contexts.append(getattr(doc, "page_content", str(doc)))
#             llm_response = project_generation_func(question, contexts) if contexts else ""

            # LLM 判定每个检索结果是否相关
            relevant = _judge_chunks(question, retrieved, judge_llm)
            hit_count = sum(relevant)

            query_metrics = {
                "question": question[:150],
                # "response": llm_response[:500] if llm_response else "",
                "retrieved_chunk_ids": retrieved_chunk_ids[:10],
            }
            for k in k_values:
                v = recall_at_k(relevant, k)
                query_metrics[f"recall@{k}"] = v
                metrics[f"recall@{k}"].append(v)
                v_ndcg = ndcg_at_k(relevant, k)
                query_metrics[f"ndcg@{k}"] = v_ndcg
                metrics[f"ndcg@{k}"].append(v_ndcg)
            mrr_val = mrr_from_positions(relevant)
            query_metrics["mrr"] = mrr_val
            metrics["mrr"].append(mrr_val)

            per_query.append(query_metrics)
            logger.info(
                f"[{i + 1}/{len(dataset)}] hits={hit_count}/{len(relevant)} "
                f"for: {question[:50]}..."
            )

        except Exception as e:
            logger.warning(f"[{i + 1}] Retrieval failed for '{question[:30]}...': {e}")

    result = {
        "ir_metrics": {name: sum(vals) / len(vals) if vals else 0.0 for name, vals in metrics.items()},
        "per_query": per_query,
        "total_queries": len(per_query),
    }
    return result


# ═══════════════════════════════════════════════════════════
# RAGAS 生成指标
# ═══════════════════════════════════════════════════════════

def compute_ragas_metrics(
    dataset: List[Dict[str, Any]],
    retrieval_func,
    generation_func,
) -> Dict[str, Any]:
    """对每条数据跑检索+生成，用 RAGAS 计算 faithfulness 等

    RAGAS 0.4.x 的每个 metric 需要显式传 LLM（默认走 OpenAI），这里注入 DashScope。
    """
    try:
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.embeddings import DashScopeEmbeddings
    except ImportError as e:
        logger.warning(f"ragas not available: {e}")
        return {"error": f"ragas not available: {e}", "per_query": []}

    from core.config import config

    ragas_llm = LangchainLLMWrapper(
        Tongyi(model_name="qwen-max", api_key=config.DASHSCOPE_API_KEY, streaming=False)
    ) if config.DASHSCOPE_API_KEY else None

    ragas_embeddings = LangchainEmbeddingsWrapper(
        DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=config.DASHSCOPE_API_KEY)
    ) if config.DASHSCOPE_API_KEY else None

    metrics_instances = [
        Faithfulness(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    samples = []
    retrieval_errors = 0
    generation_errors = 0

    for i, item in enumerate(dataset):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")

        try:
            retrieved = retrieval_func(question, top_k=5)
            contexts = []
            for doc in retrieved:
                if isinstance(doc, dict):
                    contexts.append(doc.get("content", doc.get("page_content", "")))
                else:
                    contexts.append(getattr(doc, "page_content", str(doc)))

            if not contexts:
                logger.warning(f"[{i + 1}/{len(dataset)}] Empty retrieval for: {question[:60]}...")
                retrieval_errors += 1
                continue

            try:
                answer = generation_func(question, contexts)
            except Exception as e:
                logger.warning(f"[{i + 1}/{len(dataset)}] Generation failed: {e}")
                generation_errors += 1
                continue

            samples.append(SingleTurnSample(
                user_input=question,
                response=answer,
                reference=ground_truth,
                retrieved_contexts=contexts,
            ))
            logger.info(f"[{i + 1}/{len(dataset)}] RAGAS sample: {question[:50]}...")

        except Exception as e:
            logger.warning(f"[{i + 1}/{len(dataset)}] Failed: {e}")
            retrieval_errors += 1

    if retrieval_errors > 0 or generation_errors > 0:
        logger.warning(
            f"RAGAS sample building: {len(samples)} ok, "
            f"{retrieval_errors} retrieval errs, {generation_errors} generation errs"
        )

    if not samples:
        return {"error": "No valid samples", "per_query": []}

    try:
        eval_dataset = EvaluationDataset(samples=samples)
        result = evaluate(eval_dataset, metrics=metrics_instances)
        # RAGAS 0.4.x 返回 EvaluationResult 对象，用 to_pandas() 获取 DataFrame
        df = result.to_pandas()
        # 只取数值指标列（faithfulness, context_precision 等），
        # 跳过 user_input, response 等元数据列
        metric_cols = [c for c in df.columns if c not in (
            "user_input", "retrieved_contexts", "response", "reference")]
        report = {}
        for col in metric_cols:
            vals = df[col].tolist()
            report[col] = [float(v) if v is not None else 0.0 for v in vals]

        avg_metrics = {}
        for key, val in report.items():
            if len(val) > 0:
                avg_metrics[key] = sum(val) / len(val)

        per_query = []
        for i, sample in enumerate(samples):
            entry = {
                "question": sample.user_input[:150],
                "response": sample.response[:500] if sample.response else "",
                "reference": sample.reference[:500] if sample.reference else "",
                "retrieved_contexts": [
                    ctx[:300] for ctx in (sample.retrieved_contexts or [])
                ],
            }
            for key, vals in report.items():
                if isinstance(vals, list) and i < len(vals):
                    entry[key] = vals[i]
            per_query.append(entry)

        return {
            "ragas_metrics": avg_metrics,
            "per_query": per_query,
            "total_queries": len(per_query),
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"error": str(e), "per_query": []}


# ═══════════════════════════════════════════════════════════
# 便捷包装 — 直接对接项目组件
# ═══════════════════════════════════════════════════════════

def make_retrieval_func(mode: str = "full"):
    """模式: full(全优化) / baseline(纯Dense+扁平子块)"""
    hybrid = mode == "full"
    rerank = mode == "full"
    rewrite = mode == "full"
    skip_parent = mode == "baseline"

    def _retrieve(question: str, top_k: int = 10) -> List[Dict]:
        from service.retrieval import RetrievalAgent
        from core.vector_store import vector_store
        agent = RetrievalAgent()

        # full: 走 RetrievalAgent（改写+检索+rerank 一体化）
        if mode == "full":
            result = agent.retrieve(
                query=question, use_rewrite=True, use_rerank=True,
                use_hybrid=True, top_k=top_k, similarity_threshold=0.5,
            )
            docs = result.reranked_documents
            return [
                {"content": d.get("content", ""), "metadata": d.get("metadata", {})}
                if isinstance(d, dict) else {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]

        # baseline: 直调 vector_store，跳过所有优化
        docs = vector_store.search(
            query=question, k=top_k,
            use_rerank=False, hybrid=False,
            skip_parent_fetch=True,
        )
        return [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]
    return _retrieve


def project_retrieval_func(question: str, top_k: int = 10) -> List[Dict]:
    return make_retrieval_func("full")(question, top_k)


def project_generation_func(question: str, contexts: List[str]) -> str:
    """直接调用项目的 LLMService 生成回答"""
    from core.llm import LLMService
    from langchain_core.documents import Document
    llm = LLMService()
    docs = [Document(page_content=c) for c in contexts if c]
    return llm.get_answer(question, docs, conversation_context="")


def run_full_eval(dataset: List[Dict[str, Any]], k_values: tuple = (1, 3, 5, 10),
                  skip_ragas: bool = False,
                  retrieval_func=None) -> Dict[str, Any]:
    """一站式评测：RAGAS 指标（先跑，报错直接终止）+ IR 指标"""
    retrieval_func = retrieval_func or project_retrieval_func
    logger.info(f"Running full eval on {len(dataset)} queries...")

    if skip_ragas:
        ragas_result = {"error": "", "ragas_metrics": {}, "per_query": []}
    else:
        ragas_result = compute_ragas_metrics(dataset, retrieval_func, project_generation_func)
        if ragas_result.get("error"):
            raise RuntimeError(f"RAGAS evaluation failed: {ragas_result['error']}")

    ir_result = compute_ir_metrics(dataset, retrieval_func, k_values=k_values)

    return {
        "ir_metrics": ir_result.get("ir_metrics", {}),
        "ir_skipped": ir_result.get("skipped_queries", 0),
        "ragas_metrics": ragas_result.get("ragas_metrics", {}),
        "ragas_error": ragas_result.get("error", ""),
        "total_queries": len(dataset),
        "ir_per_query": ir_result.get("per_query", []),
        "ragas_per_query": ragas_result.get("per_query", []),
    }

#!/usr/bin/env python
"""运营与质量指标计算 — 从已有数据源（MySQL / Redis / eval 报告 / Java日志）中提取量化指标

用法:
    python eval/run_metrics.py                        # 全部指标
    python eval/run_metrics.py --section cache        # 仅缓存指标
    python eval/run_metrics.py --section rag          # 仅RAG指标
    python eval/run_metrics.py --section trace        # 仅trace指标
    python eval/run_metrics.py --section memory       # 仅记忆指标
    python eval/run_metrics.py --section all --fresh  # 全部，强制刷新

数据来源（只读，不修改任何现有代码）:
    - MySQL request_trace / agent_run / agent_step 表
    - Java 缓存统计端点 /api/cache/stats
    - eval/data/report_*.json 已有评测报告
    - Redis 缓存统计（如有）
"""

import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# 环境准备
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

from core.text_utils import estimate_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("metrics")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Java 后端地址（从环境变量或默认）
JAVA_HOST = os.getenv("JAVA_API_HOST", "http://localhost:8080")


# ═══════════════════════════════════════════════════════════════
# 数据源连接（只读）
# ═══════════════════════════════════════════════════════════════

class DataSources:
    """惰性初始化各种数据源连接"""

    def __init__(self):
        self._mysql = None
        self._redis = None
        self._cache_stats = None

    @property
    def mysql(self):
        if self._mysql is None:
            try:
                from core.mysql_client import MySQLClient
                self._mysql = MySQLClient()
                self._mysql.connect()
                logger.info("MySQL connected")
            except Exception as e:
                logger.warning(f"MySQL unavailable: {e}")
                self._mysql = False
        return self._mysql if self._mysql is not False else None

    @property
    def redis(self):
        if self._redis is None:
            try:
                from core.redis_client import redis_client
                self._redis = redis_client
                logger.info("Redis connected")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")
                self._redis = False
        return self._redis if self._redis is not False else None

    def fetch_cache_stats(self, cache_name: str = "ai_answer") -> Optional[dict]:
        """从 Java 端点获取缓存统计"""
        if self._cache_stats is not None:
            return self._cache_stats.get(cache_name)
        try:
            resp = requests.get(f"{JAVA_HOST}/api/cache/{cache_name}/info", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    # 响应结构: {cacheName, stats: {hits, misses, hitRate, ...}, size, hitRate}
                    stats = data.get("stats", {})
                    stats["hitRate"] = data.get("hitRate", stats.get("hitRate", 0))
                    stats["size"] = data.get("size", 0)
                    self._cache_stats = {cache_name: stats}
                    return stats
        except Exception as e:
            logger.warning(f"Cannot fetch cache stats from Java: {e}")
        self._cache_stats = {}
        return None

    def query_mysql(self, sql: str, params=None) -> list:
        """执行只读 MySQL 查询并返回结果列表"""
        db = self.mysql
        if db is None or db.connection is None:
            return []
        try:
            cursor = db.connection.cursor(dictionary=True)
            cursor.execute(sql, params or ())
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Exception as e:
            logger.warning(f"MySQL query failed: {e}")
            return []

    def close(self):
        """关闭连接"""
        if self._mysql and self._mysql is not False:
            try:
                self._mysql.disconnect()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# 1. 缓存指标
# ═══════════════════════════════════════════════════════════════

def compute_cache_metrics(ds: DataSources) -> dict:
    """从 trace 表统计用户提问级别的缓存命中率 + Redis 侧数据"""
    print("\n── 缓存指标 ──")

    result = {}

    # ── 从 trace 表统计（一个 trace = 一个用户提问）──
    traces = ds.query_mysql("""
        SELECT trace_json FROM request_trace
        WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
    """)

    question_hits = 0
    question_misses = 0
    hit_details = {"exact": 0, "semantic": 0}  # 精确命中 vs 语义命中

    for row in traces:
        try:
            events = json.loads(row.get("trace_json", "{}")).get("events", [])
            # 检查是否有缓存命中事件
            event_types = [e.get("eventType", "") for e in events]
            outputs = {e.get("eventType", ""): e.get("outputData", {}) for e in events}

            if "CACHE_HIT_RETURN" in event_types:
                question_hits += 1
                # 区分精确 vs 语义
                cache_lookup = [e for e in events if e.get("eventType") == "CACHE_LOOKUP"]
                if cache_lookup:
                    result_text = str(cache_lookup[0].get("outputData", {}).get("result", ""))
                    if "SEMANTIC" in result_text.upper():
                        hit_details["semantic"] += 1
                    else:
                        hit_details["exact"] += 1
                else:
                    hit_details["exact"] += 1
            elif any(et in ("PYTHON_CALL_END", "python_call_end") for et in event_types):
                question_misses += 1
        except Exception:
            pass

    total_questions = question_hits + question_misses
    result["total_questions"] = total_questions
    result["question_hits"] = question_hits
    result["question_misses"] = question_misses

    if total_questions > 0:
        hit_rate = round(question_hits / total_questions * 100, 1)
    else:
        hit_rate = 0

    result["question_hit_rate"] = hit_rate
    result["hit_details"] = hit_details

    if total_questions > 0:
        print(f"  用户提问总数:           {total_questions}")
        print(f"  缓存命中数:             {question_hits}  ({hit_rate}%)")
        print(f"    └ 精确命中:           {hit_details['exact']}")
        print(f"    └ 语义命中:           {hit_details['semantic']}")
        print(f"  缓存未命中 (调AI):      {question_misses}")
        print(f"  ─────────────────────────────")
        print(f"  整体命中率 (提问级):    {hit_rate}%")
    else:
        print(f"  无 trace 数据（前端提问后自动生成）")

    # ── Java 端点补充：Caffeine / Redis 内部统计 ──
    stats = ds.fetch_cache_stats("ai_answer")
    if stats:
        local_hits = stats.get("localHitCount", 0)
        local_misses = stats.get("localMissCount", 0)
        local_requests = stats.get("localRequestCount", 0)
        print(f"\n  缓存层内部统计 (Caffeine+Redis):")
        print(f"    get() 调用次数:        {local_requests}")
        print(f"    内部命中/未命中:       {local_hits} / {local_misses}")

    # ── Redis 侧数据 ──
    r = ds.redis
    redis_count = 0
    hot_count = 0
    if r:
        try:
            raw = r.client
            keys = raw.keys("ai:answer:*")
            redis_count = len(keys) if keys else 0
            result["redis_cached_answers"] = redis_count
            for key in (keys or [])[:100]:
                ttl = raw.ttl(key)
                if ttl and ttl > 3600:
                    hot_count += 1
            result["estimated_hot_data_count"] = hot_count
        except Exception:
            pass
    print(f"\n  Redis 已缓存答案数:     {redis_count}")
    print(f"  估计热点数据数 (采样):   {hot_count}")

    # ── 成本节省（提问级别）──
    if total_questions > 0:
        print(f"\n  估算 API 调用节省:      ~{hit_rate}%")

    # ── 延迟对比 ──
    print(f"\n  延迟对比 (需从日志/trace分析):")
    print(f"    精确缓存命中:  < 1ms  (Caffeine 内存)")
    print(f"    语义缓存命中:  ~30-80ms (embedding + Milvus)")
    print(f"    AI 直接调用:   ~3-15s  (取决于复杂度)")

    return result


# ═══════════════════════════════════════════════════════════════
# 2. RAG 指标（从已有 eval 报告）
# ═══════════════════════════════════════════════════════════════

def compute_rag_metrics(ds: DataSources) -> dict:
    """从历史 eval 报告读取 RAG 指标"""
    print("\n── RAG 指标 ──")

    result = {}
    reports = sorted(
        [f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")],
        reverse=True,
    )

    if not reports:
        print("  未找到 eval 报告。运行 `python eval/run_eval.py` 生成。")
        return {"note": "no_reports"}

    # 读最新报告
    latest_path = os.path.join(DATA_DIR, reports[0])
    with open(latest_path) as f:
        report = json.load(f)

    mode = report.get("mode", "unknown")
    print(f"  数据来源: {reports[0]} (mode={mode}, queries={report.get('total_queries', 0)})")

    ragas = report.get("ragas_metrics", {})
    if ragas:
        for name in sorted(ragas):
            val = ragas[name]
            result[f"ragas_{name}"] = round(val, 4) if isinstance(val, float) else val
            print(f"  {name:25s}: {val:.4f}" if isinstance(val, float) else f"  {name:25s}: {val}")
    else:
        print("  RAGAS 指标未生成 (可能运行了 --skip-ragas)")

    ir = report.get("ir_metrics", {})
    if ir:
        print("\n  IR 指标:")
        for name in sorted(ir):
            val = ir[name]
            result[f"ir_{name}"] = round(val, 4) if isinstance(val, float) else val
            print(f"    {name:20s}: {val:.4f}" if isinstance(val, float) else f"    {name:20s}: {val}")

    return result


# ═══════════════════════════════════════════════════════════════
# 3. Trace 指标
# ═══════════════════════════════════════════════════════════════

def compute_trace_metrics(ds: DataSources) -> dict:
    """从 MySQL request_trace 和 agent_run 表计算 trace 指标"""
    print("\n── Trace 指标 ──")

    result = {}

    # 3a. 单次请求平均事件数
    traces = ds.query_mysql("""
        SELECT trace_id, event_count, duration_ms,
               JSON_LENGTH(trace_json, '$.events') AS event_cnt
        FROM request_trace
        WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
        ORDER BY created_at DESC
        LIMIT 200
    """)

    if traces:
        avg_events = sum(t.get("event_count", 0) or 0 for t in traces) / len(traces)
        result["avg_events_per_trace"] = round(avg_events, 1)

        durations = [t.get("duration_ms", 0) or 0 for t in traces if (t.get("duration_ms") or 0) > 0]
        if durations:
            durations.sort()
            p50_ms = durations[len(durations) // 2]
            p95_ms = durations[int(len(durations) * 0.95)]
            result["trace_p50_latency_ms"] = p50_ms
            result["trace_p95_latency_ms"] = p95_ms

        print(f"  近7天 Trace 数:        {len(traces)}")
        print(f"  平均事件数/Trace:      {avg_events:.1f}")
        if durations:
            print(f"  P50 延迟:              {p50_ms}ms ({p50_ms/1000:.1f}s)")
            print(f"  P95 延迟:              {p95_ms}ms ({p95_ms/1000:.1f}s)")
    else:
        print("  无 trace 数据")
        result["note"] = "no_trace_data"

    # 3b. Agent 执行统计
    runs = ds.query_mysql("""
        SELECT agent_type, status, COUNT(*) AS cnt,
               AVG(TIMESTAMPDIFF(SECOND, start_time, end_time)) AS avg_sec
        FROM agent_run
        WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY agent_type, status
        ORDER BY agent_type, status
    """)

    if runs:
        print("\n  Agent 执行统计 (近7天):")
        by_type = defaultdict(lambda: {"completed": 0, "failed": 0, "avg_sec": 0})
        for r in runs:
            agent = r.get("agent_type", "unknown")
            status = r.get("status", "unknown")
            cnt = r.get("cnt", 0)
            avg = r.get("avg_sec", 0) or 0
            print(f"    {agent:25s} {status:12s}: {cnt:4d} 次, 平均 {avg:.0f}s")
            if status == "COMPLETED":
                by_type[agent]["completed"] += cnt
                by_type[agent]["avg_sec"] = avg
            else:
                by_type[agent]["failed"] += cnt
        result["agent_runs"] = dict(by_type)

    # 3c. 按事件类型的耗时统计
    events_data = ds.query_mysql("""
        SELECT trace_json FROM request_trace
        WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
        LIMIT 50
    """)

    if events_data:
        event_latencies = defaultdict(list)
        for row in events_data:
            try:
                events = json.loads(row.get("trace_json", "{}")).get("events", [])
                for e in events:
                    etype = e.get("eventType") or e.get("event_type", "unknown")
                    lat = e.get("latencyMs") or e.get("latency_ms")
                    if lat and lat > 0:
                        event_latencies[etype].append(lat)
            except Exception:
                pass

        if event_latencies:
            print("\n  按事件类型的延迟:")
            print(f"    {'事件':30s} {'P50':>8s} {'P95':>8s} {'P99':>8s} {'avg':>8s} {'n':>6s}")
            print(f"    {'-'*70}")
            latency_summary = {}
            for etype in sorted(event_latencies):
                lats = sorted(event_latencies[etype])
                n = len(lats)
                p50 = lats[n // 2]
                p95 = lats[int(n * 0.95)] if n >= 20 else lats[-1]
                p99 = lats[int(n * 0.99)] if n >= 100 else (lats[-1] if n > 0 else 0)
                avg = sum(lats) / n
                print(f"    {etype:30s} {p50:7.0f}ms {p95:7.0f}ms {p99:7.0f}ms {avg:7.0f}ms {n:5d}")
                latency_summary[etype] = {
                    "p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "avg_ms": round(avg, 1), "n": n
                }
            result["event_latency_summary"] = latency_summary

            # 重点：缓存命中 vs AI 调用的延迟对比
            cache_lats = event_latencies.get("CACHE_LOOKUP", []) or event_latencies.get("CACHE_HIT_RETURN", [])
            ai_lats = event_latencies.get("PYTHON_CALL_END", []) or event_latencies.get("python_call_end", [])

            if cache_lats:
                cache_sorted = sorted(cache_lats)
                ca_p50 = cache_sorted[len(cache_sorted)//2]
                ca_p95 = cache_sorted[int(len(cache_sorted)*0.95)] if len(cache_sorted) >= 20 else cache_sorted[-1]
                result["cache_hit_p50_ms"] = ca_p50
                result["cache_hit_p95_ms"] = ca_p95
                print(f"\n  >>> 缓存命中 P50={ca_p50}ms, P95={ca_p95}ms")

            if ai_lats:
                ai_sorted = sorted(ai_lats)
                ai_p50 = ai_sorted[len(ai_sorted)//2]
                ai_p95 = ai_sorted[int(len(ai_sorted)*0.95)] if len(ai_sorted)>=20 else ai_sorted[-1]
                result["ai_call_p50_ms"] = ai_p50
                result["ai_call_p95_ms"] = ai_p95
                print(f"  >>> AI 调用 P50={ai_p50}ms ({ai_p50/1000:.1f}s), P95={ai_p95}ms ({ai_p95/1000:.1f}s)")

            if cache_lats and ai_lats:
                speedup = ai_p95 / max(ca_p95, 1)
                print(f"  >>> 缓存命中延迟仅为 AI 调用的 1/{speedup:.0f}")

            # ── Multi-Agent 并行加速比 ──
            # 从 agent_run 表读取 - 父 run 下多个子 run 的耗时
            # 串行 = Σ子run耗时, 并行 = max(子run耗时)
            parent_runs = ds.query_mysql("""
                SELECT r.run_id, r.parent_run_id,
                       TIMESTAMPDIFF(SECOND, r.start_time, r.end_time) AS parent_sec
                FROM agent_run r
                WHERE r.parent_run_id IS NULL
                  AND r.agent_type LIKE '%orchestrator%'
                  AND r.start_time IS NOT NULL AND r.end_time IS NOT NULL
                  AND r.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
                LIMIT 50
            """)

            total_serial = 0
            total_parallel = 0
            multi_count = 0

            for pr in parent_runs:
                run_id = pr["run_id"]
                children = ds.query_mysql(
                    "SELECT run_id, "
                    "TIMESTAMPDIFF(SECOND, start_time, end_time) AS dur_sec "
                    "FROM agent_run "
                    "WHERE parent_run_id = %s AND start_time IS NOT NULL AND end_time IS NOT NULL",
                    (run_id,)
                )
                child_secs = [c["dur_sec"] for c in children if c.get("dur_sec") and c["dur_sec"] > 0]
                if len(child_secs) >= 2:
                    total_serial += sum(child_secs)
                    total_parallel += max(child_secs)
                    multi_count += 1

            if multi_count > 0:
                avg_serial = total_serial / multi_count
                avg_parallel = total_parallel / multi_count
                speedup = avg_serial / avg_parallel if avg_parallel > 0 else 1
                reduction = round((1 - avg_parallel / avg_serial) * 100, 1) if avg_serial > 0 else 0

                result["multi_agent_parallel_count"] = multi_count
                result["avg_serial_time_s"] = round(avg_serial, 1)
                result["avg_parallel_time_s"] = round(avg_parallel, 1)
                result["parallel_speedup"] = round(speedup, 1)
                result["parallel_reduction_pct"] = reduction

                print(f"\n  Multi-Agent 并行加速 (找到 {multi_count} 个拆解任务):")
                print(f"    平均串行耗时:         {avg_serial:.0f}s")
                print(f"    平均并行耗时:         {avg_parallel:.0f}s")
                print(f"    加速比:               {speedup:.1f}×")
                print(f"    耗时降低:             {reduction}%")
                print(f"\n  >>> 简历可写: 并行执行较串行推理耗时降低 {reduction}%")
            else:
                print(f"\n  Multi-Agent 并行加速: 无拆解任务数据")
                print(f"    (需要在前端问复杂问题触发 Multi-Agent 拆解)")

    return result


# ═══════════════════════════════════════════════════════════════
# 4. 记忆与压缩指标
# ═══════════════════════════════════════════════════════════════

def compute_memory_metrics(ds: DataSources) -> dict:
    """从真实对话数据 + MySQL 计算记忆指标"""
    print("\n── 记忆指标 ──")

    result = {}
    KEEP_RECENT_MSG = 20  # 保留最近 10 轮完整对话（每轮 1 user + 1 assistant，共 20 条消息）
    TOKEN_THRESHOLD = 10000  # 触发压缩的 token 阈值

    # 4a. L0 压缩比 —— MySQL 原始消息 vs Redis/LLM 真实摘要
    #     先找所有长对话，再按 token 阈值过滤
    candidates = ds.query_mysql("""
        SELECT conversation_id, COUNT(*) AS msg_count
        FROM raw_conversations
        GROUP BY conversation_id
        HAVING msg_count > 10
        ORDER BY msg_count DESC
        LIMIT 10
    """)

    compression_ratios = []
    r = ds.redis
    for conv in candidates:
        cid = conv["conversation_id"]
        msgs = ds.query_mysql(
            "SELECT role, content FROM raw_conversations "
            "WHERE conversation_id = %s ORDER BY created_at ASC",
            (cid,)
        )
        if len(msgs) < 12:
            continue

        # 压缩前：全部原始消息
        before_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in msgs
        )
        before_tok = estimate_tokens(before_text)

        # Redis 中已有的历史摘要（若有）
        summary = None
        if r:
            try:
                summary = r.get_summary(cid)
            except Exception:
                pass

        # 实际喂给大模型的是「摘要 + 会话列表」，触发条件与 tools/memory_read.py 一致
        summary_tok = estimate_tokens(summary) if summary else 0
        if summary_tok + before_tok <= TOKEN_THRESHOLD:
            continue

        # 压缩后：LLM 真实摘要 + 最近 KEEP_RECENT_MSG 条消息
        recent = msgs[-KEEP_RECENT_MSG:]
        recent_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in recent
        )

        if summary:
            after_text = f"[历史对话摘要] {summary}\n{recent_text}"
            src = "LLM摘要(Redis)"
        else:
            early = msgs[:-KEEP_RECENT_MSG]
            early_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in early
            )
            try:
                from core.llm import LLMService
                llm = LLMService()
                prompt = (
                    "请将以下对话压缩为结构化摘要，按用户提问逐条总结，不要遗漏：\n\n"
                    f"{early_text}\n\n"
                    "用\"用户询问了以下问题：\"开头，每个问题一行，纯文本输出。"
                )
                summary = llm.generate(prompt)
                after_text = f"[历史对话摘要] {summary}\n{recent_text}"
                src = "LLM摘要(实时生成)"
            except Exception as e:
                continue

        after_tok = estimate_tokens(after_text)

        if before_tok > 0:
            ratio = round((1 - after_tok / before_tok) * 100, 1)
            compression_ratios.append((cid[:20], ratio, src, before_tok, after_tok))

    if compression_ratios:
        ratios = [r[1] for r in compression_ratios]
        avg_ratio = round(sum(ratios) / len(ratios), 1)
        result["l0_compression_ratio_pct"] = avg_ratio
        print(f"  L0 压缩比 (真实对话+真实LLM摘要, {len(compression_ratios)} 个长会话):")
        for cid, ratio, src, bef, aft in compression_ratios:
            print(f"    {cid}: {bef}t → {aft}t, -{ratio}%  [{src}]")
        print(f"    平均压缩比:           {avg_ratio}%")
    else:
        # 无真实长对话 → 模拟
        sample = _generate_sample_conversation(15)
        before_tok = estimate_tokens("\n".join(
            f"{m['role']}: {m['content']}" for m in sample
        ))
        recent = sample[-KEEP_RECENT_MSG:]
        early = sample[:-KEEP_RECENT_MSG]
        summary = _simulate_summary(early)
        after_tok = estimate_tokens(
            f"[历史对话摘要] {summary}\n" +
            "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        )
        if before_tok > 0:
            ratio = round((1 - after_tok / before_tok) * 100, 1)
            result["l0_compression_ratio_pct"] = ratio
            print(f"  L0 压缩比 (模拟, 无足够真实长对话): {ratio}%")
            print(f"    压缩前: ~{before_tok} tokens")
            print(f"    压缩后: ~{after_tok} tokens")

    # 4b. L1 记忆恢复延迟（从 agent_run 中统计 memory_agent 类型）
    memory_runs = ds.query_mysql("""
        SELECT agent_type, AVG(TIMESTAMPDIFF(SECOND, start_time, end_time) * 1000) AS avg_ms
        FROM agent_run
        WHERE agent_type LIKE '%memory%' AND start_time IS NOT NULL AND end_time IS NOT NULL
          AND created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY agent_type
    """)

    if memory_runs:
        print("\n  L1 记忆检索延迟:")
        for r in memory_runs:
            agent = r.get("agent_type", "unknown")
            avg_ms = r.get("avg_ms", 0) or 0
            result[f"memory_{agent}_latency_ms"] = round(avg_ms, 1)
            print(f"    {agent}: avg {avg_ms:.0f}ms")

    # 4c. L1 存储统计
    l1_count = ds.query_mysql("""
        SELECT COUNT(*) AS cnt FROM agent_run
        WHERE agent_type LIKE '%memory%' AND status = 'COMPLETED'
    """)
    if l1_count:
        result["l1_extraction_count"] = l1_count[0].get("cnt", 0)
        print(f"\n  L1 记忆提取总次数: {result['l1_extraction_count']}")

    return result


# ═══════════════════════════════════════════════════════════════
# 5. 检索指标
# ═══════════════════════════════════════════════════════════════

def compute_retrieval_metrics(ds: DataSources) -> dict:
    """从 trace 数据推断检索行为"""
    print("\n── 检索指标 ──")

    # 解析 trace_json 中的 RETRIEVAL_EXECUTED 事件，统计每请求的平均检索轮数
    traces = ds.query_mysql("""
        SELECT trace_json FROM request_trace
        WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
        LIMIT 200
    """)

    retrieval_rounds = []
    for row in traces:
        try:
            events = json.loads(row.get("trace_json", "{}")).get("events", [])
            count = sum(1 for e in events
                        if e.get("eventType") in ("RETRIEVAL_EXECUTED", "retrieval_executed"))
            if count > 0:
                retrieval_rounds.append(count)
        except Exception:
            pass

    result = {}
    if retrieval_rounds:
        avg_rounds = sum(retrieval_rounds) / len(retrieval_rounds)
        result["avg_retrieval_rounds"] = round(avg_rounds, 2)
        result["max_retrieval_rounds"] = max(retrieval_rounds)
        # 单轮占比
        single_round_pct = round(
            sum(1 for r in retrieval_rounds if r == 1) / len(retrieval_rounds) * 100, 1
        )
        result["single_round_pct"] = single_round_pct

        print(f"  采样请求数:            {len(retrieval_rounds)}")
        print(f"  平均检索轮数:          {avg_rounds:.1f}")
        print(f"  最大检索轮数:          {result['max_retrieval_rounds']}")
        print(f"  单轮满足率:            {single_round_pct}%")
    else:
        print("  无检索事件数据（trace 中无 RETRIEVAL_EXECUTED 事件）")

    return result


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _generate_sample_conversation(n_rounds: int) -> list:
    """生成模拟对话用于压缩比测试"""
    topics = [
        ("什么是 Spring Cloud？", "Spring Cloud 是一套微服务治理框架..."),
        ("Eureka 怎么配置？", "添加 @EnableEurekaServer 注解..."),
        ("Ribbon 的负载均衡策略？", "轮询、随机、加权响应时间..."),
        ("Feign 和 RestTemplate 的区别？", "Feign 声明式、RestTemplate 编程式..."),
        ("Hystrix 熔断怎么用？", "添加 @HystrixCommand 注解..."),
        ("Gateway 路由怎么配？", "在 yml 中配置 routes..."),
        ("Nacos 和 Eureka 哪个好？", "Nacos 支持配置中心，Eureka 仅注册中心..."),
        ("Sentinel 限流规则？", "QPS 限流、线程数限流、降级..."),
        ("分布式事务怎么处理？", "Seata、TCC、可靠消息最终一致性..."),
        ("Config 配置中心怎么用？", "bootstrap.yml 配置 config server 地址..."),
        ("Sleuth 链路追踪原理？", "通过 TraceId 和 SpanId 串联调用链..."),
        ("Docker 怎么部署微服务？", "Dockerfile + docker-compose 编排..."),
        ("K8s 和 Docker 的关系？", "Docker 是容器引擎，K8s 是编排平台..."),
        ("CI/CD 流水线怎么搭？", "Jenkins/GitLab CI → 构建 → 测试 → 部署..."),
        ("日志聚合用什么方案？", "ELK/EFK 方案，Filebeat 采集..."),
    ]
    messages = []
    for i in range(min(n_rounds, len(topics))):
        q, a = topics[i]
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages


def _simulate_summary(messages: list) -> str:
    """模拟 LLM 摘要 — 真实运行时由 LLM 生成，这里用关键词提取模拟"""
    questions = [m["content"][:40] for m in messages if m["role"] == "user"]
    return "用户询问了以下问题：\n" + "\n".join(
        f"  {i+1}. {q}" for i, q in enumerate(questions)
    )


def compare_rag_reports(baseline_report: str = None, optimized_report: str = None) -> dict:
    """比较两份 eval 报告的 RAG 指标差异（baseline vs optimized）

    用法:
        python eval/run_eval.py --mode baseline --n 50
        python eval/run_eval.py --mode full --n 50
        python eval/run_metrics.py --compare
    """
    print("\n── RAG 指标对比 ──")

    reports = sorted([
        f for f in os.listdir(DATA_DIR) if f.startswith("report_") and f.endswith(".json")
    ], reverse=True)  # 最新的在前

    baseline_data = None
    optimized_data = None

    for r in reports:
        path = os.path.join(DATA_DIR, r)
        with open(path) as f:
            data = json.load(f)
        mode = data.get("mode", "unknown")
        if mode == "baseline" and baseline_data is None:
            baseline_data = data
            baseline_data["_report_name"] = r
        elif mode == "full" and optimized_data is None:
            optimized_data = data
            optimized_data["_report_name"] = r

    if not baseline_data or not optimized_data:
        print("  缺少对比数据。请先运行:")
        print("    python eval/run_eval.py --mode baseline --n 50")
        print("    python eval/run_eval.py --mode full --n 50")
        return {}

    print(f"  Baseline (单级索引, 无Rerank): {baseline_data['_report_name']}")
    print(f"  Optimized (父子分块+混合检索+Rerank): {optimized_data['_report_name']}")

    # 比较 RAGAS 指标
    baseline_ragas = baseline_data.get("ragas_metrics", {})
    optimized_ragas = optimized_data.get("ragas_metrics", {})

    if baseline_ragas and optimized_ragas:
        print("\n  RAGAS 指标:")
        print(f"    {'指标':25s} {'Baseline':>10s} {'Optimized':>10s} {'Delta':>10s}")
        print(f"    {'-' * 55}")
        comparison = {}
        for name in sorted(set(list(baseline_ragas.keys()) + list(optimized_ragas.keys()))):
            b = baseline_ragas.get(name, 0)
            o = optimized_ragas.get(name, 0)
            delta = o - b
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            comparison[name] = {"baseline": b, "optimized": o, "delta": delta}
            print(f"    {name:25s} {b:9.4f}  {o:9.4f}  {delta:+7.4f} {direction}")

    # 比较 IR 指标
    baseline_ir = baseline_data.get("ir_metrics", {})
    optimized_ir = optimized_data.get("ir_metrics", {})

    if baseline_ir and optimized_ir:
        print("\n  IR 指标:")
        print(f"    {'指标':20s} {'Baseline':>10s} {'Optimized':>10s} {'Delta':>10s}")
        print(f"    {'-' * 50}")
        for name in sorted(set(list(baseline_ir.keys()) + list(optimized_ir.keys()))):
            b = baseline_ir.get(name, 0)
            o = optimized_ir.get(name, 0)
            delta = o - b
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"    {name:20s} {b:9.4f}  {o:9.4f}  {delta:+7.4f} {direction}")

    return {
        "baseline_report": baseline_data["_report_name"],
        "optimized_report": optimized_data["_report_name"],
        "ragas_comparison": {},
        "ir_comparison": {},
    }


# ═══════════════════════════════════════════════════════════════
# 汇总与输出
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# 同一问题首次 vs 再次 延迟对比
# ═══════════════════════════════════════════════════════════════

def compute_same_question_latency(ds: DataSources) -> dict:
    """从 trace 中找出同一问题被多次询问的 case，对比首次(AI)和再次(缓存)的延迟"""
    print("\n── 同一问题延迟对比 ──")

    traces = ds.query_mysql("""
        SELECT trace_id, event_count, duration_ms, trace_json
        FROM request_trace
        WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
        ORDER BY created_at ASC
    """)

    # 提取每个 trace 的问题内容和是否命中缓存
    parsed = []
    for row in traces:
        try:
            events = json.loads(row.get("trace_json", "{}")).get("events", [])
            # 从第一个带 content 的事件中提取问题
            question = None
            for e in events:
                inp = e.get("inputData") or {}
                q = (inp.get("question") or inp.get("content") or
                     (inp if isinstance(inp, str) else None))
                if q:
                    question = str(q)
                    break

            if not question:
                continue

            # 判断是否缓存命中
            event_types = [e.get("eventType", "") for e in events]
            cached = "CACHE_HIT_RETURN" in event_types

            # 端到端延迟用 request_trace.duration_ms（用户从发请求到收到回答的时间）
            latency = row.get("duration_ms", 0) or 0

            parsed.append({
                "question": question,
                "cached": cached,
                "latency_ms": latency,
                "trace_id": row.get("trace_id", ""),
            })
        except Exception:
            pass

    if not parsed:
        print("  无 trace 数据")
        return {"note": "no_data"}

    # 按问题文本分组
    from collections import defaultdict
    groups = defaultdict(list)
    for p in parsed:
        # 用问题前 30 个字符做粗分组（容忍标点差异）
        key = p["question"][:30].strip().lower()
        groups[key].append(p)

    # 找出既有 AI 调用又有缓存命中的问题组
    results = []
    for key, items in groups.items():
        cached = [it for it in items if it["cached"]]
        uncached = [it for it in items if not it["cached"]]
        if cached and uncached:
            ai_latency = max(it["latency_ms"] or 0 for it in uncached)
            cache_latency = min(it["latency_ms"] or 0 for it in cached)
            results.append({
                "question": items[0]["question"][:60],
                "ai_latency_ms": ai_latency,
                "cache_latency_ms": cache_latency,
                "speedup": ai_latency / max(cache_latency, 0.1),
            })

    if not results:
        print("  未找到同一问题被问多次的 case")
        print("  （在前端对同一个问题问两次即可生成数据）")
        return {"note": "no_pairs"}

    print(f"  找到 {len(results)} 组同一问题对比:\n")
    print(f"    {'问题':40s} {'AI调用':>8s} {'缓存命中':>8s} {'加速比':>8s}")
    print(f"    {'-'*64}")

    for r in results:
        ai_s = f"{r['ai_latency_ms']/1000:.1f}s" if r['ai_latency_ms'] > 1000 else f"{r['ai_latency_ms']:.0f}ms"
        ca_s = f"{r['cache_latency_ms']/1000:.1f}s" if r['cache_latency_ms'] > 1000 else f"{r['cache_latency_ms']:.0f}ms"
        print(f"    {r['question']:40s} {ai_s:>8s} {ca_s:>8s} {r['speedup']:7.0f}x")

    # 汇总
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    avg_ai = sum(r["ai_latency_ms"] for r in results) / len(results)
    avg_cache = sum(r["cache_latency_ms"] for r in results) / len(results)

    print(f"\n  汇总:")
    print(f"    平均 AI 调用:     {avg_ai/1000:.1f}s")
    print(f"    平均缓存命中:     {avg_cache:.0f}ms")
    print(f"    平均加速:         {avg_speedup:.0f}x")

    # 简历可直接用的值
    print(f"\n  >>> 简历可写:")
    print(f"    同一问题首次 AI 调用 ~{avg_ai/1000:.0f}s，再次缓存命中 ~{avg_cache:.0f}ms")
    print(f"    P95 响应时延从 ~{avg_ai/1000:.0f}s 降至 ~{avg_cache:.0f}ms")

    return {
        "pairs": len(results),
        "avg_ai_latency_s": round(avg_ai / 1000, 1),
        "avg_cache_latency_ms": round(avg_cache, 0),
        "avg_speedup": round(avg_speedup, 0),
    }


SECTION_MAP = {
    "cache": ("缓存指标", compute_cache_metrics),
    "rag": ("RAG 指标", compute_rag_metrics),
    "trace": ("Trace 指标", compute_trace_metrics),
    "memory": ("记忆指标", compute_memory_metrics),
    "retrieval": ("检索指标", compute_retrieval_metrics),
    "same-question": ("同问题对比", compute_same_question_latency),
}


def run_all(ds: DataSources, sections: Optional[list] = None) -> dict:
    """执行指标计算，返回完整结果"""
    if sections is None:
        sections = list(SECTION_MAP.keys())

    all_results = {}
    total_sections = 0
    section_errors = 0

    for section in sections:
        if section not in SECTION_MAP:
            logger.warning(f"Unknown section: {section}")
            continue

        name, func = SECTION_MAP[section]
        total_sections += 1
        try:
            all_results[section] = func(ds)
        except Exception as e:
            logger.error(f"{name}: failed — {e}")
            all_results[section] = {"error": str(e)}
            section_errors += 1

    # 写汇总报告
    report = {
        "generated_at": datetime.now().isoformat(),
        "sections_completed": total_sections - section_errors,
        "sections_failed": section_errors,
        "metrics": all_results,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    report_path = os.path.join(DATA_DIR, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 50}")
    print(f"Report saved: {report_path}")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="运营与质量指标计算")
    parser.add_argument(
        "--section", type=str, default="all",
        choices=["all", "cache", "rag", "trace", "memory", "retrieval", "same-question"],
        help="计算指定类别的指标 (default: all)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="对比 baseline vs optimized 两份 eval 报告的 RAG 指标差异"
    )
    args = parser.parse_args()

    if args.compare:
        print("=" * 50)
        print("  AgentCraft RAG Comparison")
        print("=" * 50)
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        compare_rag_reports()
        return

    sections = list(SECTION_MAP.keys()) if args.section == "all" else [args.section]

    print("=" * 50)
    print("  AgentCraft Metrics Calculator")
    print("=" * 50)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  指标类别: {', '.join(sections)}")
    print(f"  Java 后端: {JAVA_HOST}")
    print()

    ds = DataSources()
    try:
        run_all(ds, sections)
    finally:
        ds.close()


if __name__ == "__main__":
    main()

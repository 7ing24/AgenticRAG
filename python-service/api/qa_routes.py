from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core.llm import LLMService
from core.config import config
from core.redis_client import redis_client
from agent.router_agent import RouterAgent
import logging
import json
import time

logger = logging.getLogger(__name__)

router = APIRouter()

router_agent = RouterAgent()
llm_service = LLMService()

# ── 知识问答答案缓存（Redis 精确 + Milvus 语义两层） ────────────────
_CACHE_PREFIX = "qa:answer:"


def _cache_key(question: str) -> str:
    return _CACHE_PREFIX + (question or "").strip()


def _lookup_cached_answer(question: str):
    """两级缓存查找：先 Redis 精确命中，未命中再走 Milvus 语义找相似问题。

    Returns: (answer, cache_key)；均 None 表示未命中。
    """
    if not config.QA_CACHE_ENABLED:
        return None, None

    key = _cache_key(question)
    hit = redis_client.get_cache(key)
    if hit is not None:
        return hit, key

    # 语义命中 → 用匹配到的历史问题文本作为 Redis key 再查一次精确缓存
    try:
        from service.semantic_cache import semantic_cache_store
        found, matched_q, _sim = semantic_cache_store.lookup(question)
        if found and matched_q:
            matched_key = _cache_key(matched_q)
            hit = redis_client.get_cache(matched_key)
            if hit is not None:
                return hit, matched_key
    except Exception as e:
        logger.warning(f"Semantic cache lookup failed, skip: {e}")
    return None, None


def _cache_answer(question: str, answer: str, task_type: str, is_error: bool = False):
    """仅缓存知识问答答案：写 Redis 精确缓存，并同步问题进 Milvus 语义索引。"""
    if not config.QA_CACHE_ENABLED:
        return
    if task_type != "knowledge_qa" or not answer or is_error:
        return
    try:
        redis_client.set_cache(_cache_key(question), answer, ttl=config.QA_CACHE_TTL)
    except Exception as e:
        logger.warning(f"Cache answer (redis) failed: {e}")
        return
    try:
        from service.semantic_cache import semantic_cache_store
        semantic_cache_store.add_question(question)
    except Exception as e:
        logger.warning(f"Cache answer (semantic index) failed: {e}")


class ChatRequest(BaseModel):
    question: str
    conversation_id: str = None  # Optional, for conversation memory
    username: str = None  # Optional, if username is provided
    user_id: str = None  # Optional, user ID for user profile
    is_admin: bool = False  # Optional, whether user is admin
    trace_id: str = None  # Optional, full-chain tracing ID


class SummaryRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask_question(request: ChatRequest):
    """
    问答接口 - 使用 RouterAgent 进行任务路由
    """
    start_time = time.time()
    try:
        logger.info(f"Received question: {request.question}, username: {request.username}, is_admin: {request.is_admin}")

        # 处理身份相关问题
        lower_question = request.question.lower()
        identity_keywords = ["我是谁", "我叫什么", "我的名字", "我的身份"]
        if any(keyword in lower_question for keyword in identity_keywords) and request.username:
            logger.info(f"Answering identity question for user: {request.username}")
            answer = f"你是 {request.username}，是本系统的注册用户。"
            response = {"answer": answer, "sources": [], "task_type": "chitchat"}
        else:
            # 缓存查找（知识问答答案两级缓存：Redis 精确 + Milvus 语义）
            cached, cache_key = _lookup_cached_answer(request.question)
            if cached is not None:
                logger.info(f"Cache HIT for question: {request.question[:50]}... (key={cache_key[:50]}...)")
                response = {
                    "answer": cached,
                    "sources": [],
                    "has_sources": False,
                    "task_type": "knowledge_qa",
                    "cached": True,
                }
            else:
                # 使用 RouterAgent 进行任务路由
                result = router_agent.route(
                    input_text=request.question,
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    username=request.username,
                    is_admin=request.is_admin,
                    trace_id=request.trace_id or "",
                )

                # 构建响应
                response = {
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "task_type": result.get("task_type", "unknown"),
                    "steps": result.get("steps", []),
                    "trace_id": result.get("trace_id", request.trace_id or ""),
                    "runs": result.get("runs", []),
                    "traces": result.get("traces", []),
                }

                # 仅知识问答答案写缓存（Redis 精确 + Milvus 语义索引），错误响应不缓存
                _cache_answer(request.question, result.get("answer", ""),
                              result.get("task_type", ""),
                              is_error=result.get("error", False))

        logger.info(f"Response generated successfully, task_type: {response.get('task_type')}")

        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "POST", "path": "/api/ask",
            "status_code": 200, "process_time": process_time
        }))
        return response
    except HTTPException as e:
        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "POST", "path": "/api/ask",
            "status_code": e.status_code, "process_time": process_time
        }))
        raise
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error processing question: {str(e)}")
        logger.info(json.dumps({
            "method": "POST", "path": "/api/ask",
            "status_code": 500, "process_time": process_time
        }))
        raise HTTPException(status_code=500, detail="问答处理失败")


@router.post("/ask/stream")
async def ask_question_stream(request: ChatRequest):
    """
    流式问答接口 (Server-Sent Events) - 使用 RouterAgent
    """
    start_time = time.time()

    async def event_generator():
        try:
            logger.info(f"Streaming question: {request.question}, username: {request.username}, is_admin: {request.is_admin}")

            # 处理身份相关问题
            lower_question = request.question.lower()
            identity_keywords = ["我是谁", "我叫什么", "我的名字", "我的身份"]
            if any(keyword in lower_question for keyword in identity_keywords) and request.username:
                logger.info(f"Streaming identity answer for user: {request.username}")
                answer = f"你是 {request.username}，是本系统的注册用户。"
                for char in answer:
                    yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
                yield f"data: {json.dumps({'type': 'end', 'content': answer, 'task_type': 'chitchat'})}\n\n"
                return

            # 缓存查找：命中则回放 routed + token + end 事件（保持与真实流一致的 SSE 契约）
            cached, _cache_key_ = _lookup_cached_answer(request.question)
            if cached is not None:
                logger.info(f"Cache HIT (stream) for question: {request.question[:50]}...")
                yield f"data: {json.dumps({'type': 'routed', 'task_type': 'knowledge_qa'})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': cached})}\n\n"
                yield f"data: {json.dumps({'type': 'end', 'content': cached, 'task_type': 'knowledge_qa'})}\n\n"
                return

            # 使用 RouterAgent 进行流式任务路由
            cached_task_type = {"value": None}
            for event_data in router_agent.route_stream(
                input_text=request.question,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                username=request.username,
                is_admin=request.is_admin,
                trace_id=request.trace_id or "",
            ):
                yield f"data: {event_data}\n\n"
                try:
                    ev = json.loads(event_data)
                except Exception:
                    continue
                if ev.get("type") == "end":
                    cached_task_type["value"] = ev.get("task_type")
                    cached_task_type["answer"] = ev.get("content", "")

            # 仅知识问答答案写缓存，错误响应不缓存
            if cached_task_type.get("value") == "knowledge_qa":
                _cache_answer(request.question, cached_task_type.get("answer", ""), "knowledge_qa",
                              is_error=cached_task_type.get("error", False))

            process_time = time.time() - start_time
            logger.info(f"Stream response completed successfully in {process_time:.2f}s")
            logger.info(json.dumps({
                "method": "POST", "path": "/api/ask/stream",
                "status_code": 200, "process_time": process_time
            }))

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Error in streaming question: {str(e)}")
            logger.info(json.dumps({
                "method": "POST", "path": "/api/ask/stream",
                "status_code": 500, "process_time": process_time
            }))
            yield f"data: {json.dumps({'type': 'error', 'content': '流式问答处理失败'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
        }
    )


@router.post("/summary")
async def generate_summary(request: SummaryRequest):
    """
    生成会话标题
    """
    start_time = time.time()
    try:
        title = llm_service.generate_title(request.question)
        logger.info(f"Generated summary: {title}")

        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "POST", "path": "/api/summary",
            "status_code": 200, "process_time": process_time
        }))
        return {"title": title}
    except HTTPException as e:
        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "POST", "path": "/api/summary",
            "status_code": e.status_code, "process_time": process_time
        }))
        raise
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error generating summary: {str(e)}")
        logger.info(json.dumps({
            "method": "POST", "path": "/api/summary",
            "status_code": 500, "process_time": process_time
        }))
        raise HTTPException(status_code=500, detail="标题生成失败")

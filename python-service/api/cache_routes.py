from fastapi import APIRouter
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class CacheAddRequest(BaseModel):
    question: str


class CacheLookupRequest(BaseModel):
    question: str


@router.post("/cache/question")
async def add_to_semantic_cache(request: CacheAddRequest):
    """向语义缓存索引中添加问题（用于缓存写入时）"""
    from service.semantic_cache import semantic_cache_store
    semantic_cache_store.add_question(request.question)
    return {"status": "ok"}


@router.post("/cache/lookup")
async def semantic_cache_lookup(request: CacheLookupRequest):
    """语义缓存查找（用于精确匹配未命中时）"""
    from service.semantic_cache import semantic_cache_store
    found, cache_key, similarity = semantic_cache_store.lookup(request.question)
    return {
        "found": found,
        "cache_key": cache_key,
        "similarity": round(similarity, 4),
    }

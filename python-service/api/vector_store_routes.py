from fastapi import APIRouter, HTTPException
from core.vector_store import vector_store
import logging
import json
import time

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/vector-store/stats")
async def get_vector_store_stats():
    """
    获取向量库统计信息
    """
    start_time = time.time()
    try:
        stats = vector_store.get_stats()

        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "GET", "path": "/api/vector-store/stats",
            "status_code": 200, "process_time": process_time
        }))
        return stats
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error getting vector store stats: {str(e)}")
        logger.info(json.dumps({
            "method": "GET", "path": "/api/vector-store/stats",
            "status_code": 500, "process_time": process_time
        }))
        raise HTTPException(status_code=500, detail="获取向量库统计信息失败")


@router.delete("/vector-store/collection")
async def delete_vector_collection():
    """
    删除整个向量库（慎用）
    """
    start_time = time.time()
    try:
        vector_store.delete_collection()

        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "DELETE", "path": "/api/vector-store/collection",
            "status_code": 200, "process_time": process_time
        }))
        return {"status": "success", "message": "向量库已删除"}
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error deleting vector collection: {str(e)}")
        logger.info(json.dumps({
            "method": "DELETE", "path": "/api/vector-store/collection",
            "status_code": 500, "process_time": process_time
        }))
        raise HTTPException(status_code=500, detail="删除向量库失败")
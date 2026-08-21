import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from service.parser import DocumentParser
from core.vector_store import vector_store
import os
import logging
import json
import time

logger = logging.getLogger(__name__)

router = APIRouter()

parser = DocumentParser()


class ParseRequest(BaseModel):
    file_path: str
    doc_id: int
    trace_id: str = ""  # 全链路追踪 ID


@router.post("/parse")
async def parse_document(request: ParseRequest):
    """
    解析文档并存入向量库
    """
    start_time = time.time()
    try:
        # 判断 file_path 是 URL 还是本地路径
        is_url = request.file_path.startswith('http://') or request.file_path.startswith('https://')

        if not is_url and not os.path.exists(request.file_path):
            logger.warning(f"File not found: {request.file_path}")
            raise HTTPException(status_code=404, detail="文件不存在")

        logger.info(f"Parsing document: {request.file_path}")
        # 父子块模式: 先加载原始页（设置 doc_id）→ 再双切分
        raw_docs = await asyncio.to_thread(parser.parse, request.file_path, skip_chunk=True)
        for doc in raw_docs:
            doc.metadata["doc_id"] = request.doc_id
            doc.metadata["source"] = request.file_path
        parent_docs, child_docs = await asyncio.to_thread(parser.parent_child_chunker.split_documents, raw_docs)

        logger.info(
            f"Generated {len(parent_docs)} parents + {len(child_docs)} children. "
            f"Adding to vector store..."
        )
        try:
            await asyncio.to_thread(vector_store.add_documents, child_docs, parent_documents=parent_docs)
        except Exception as ve:
            logger.error(f"Vector store add_documents failed: {type(ve).__name__}: {ve}", exc_info=True)
            raise ve

        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "POST", "path": "/api/parse",
            "status_code": 200, "process_time": process_time
        }))
        return {"status": "success", "chunks_count": len(child_docs)}
    except HTTPException as e:
        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "POST", "path": "/api/parse",
            "status_code": e.status_code, "process_time": process_time
        }))
        raise
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error parsing document: {type(e).__name__}: {str(e)}", exc_info=True)
        logger.info(json.dumps({
            "method": "POST", "path": "/api/parse",
            "status_code": 500, "process_time": process_time
        }))
        raise HTTPException(status_code=500, detail="文档解析失败")


@router.post("/delete")
async def delete_document(request: ParseRequest):
    """
    删除文档的向量索引
    """
    start_time = time.time()
    try:
        if request.doc_id:
            logger.info(f"Deleting document with doc_id: {request.doc_id}")
            await asyncio.to_thread(vector_store.delete_document, request.doc_id)

            process_time = time.time() - start_time
            logger.info(json.dumps({
                "method": "POST", "path": "/api/delete",
                "status_code": 200, "process_time": process_time
            }))
            return {"status": "success", "message": f"Document {request.doc_id} deleted"}
        else:
            logger.warning("doc_id is required")
            raise HTTPException(status_code=400, detail="doc_id is required")
    except HTTPException as e:
        process_time = time.time() - start_time
        logger.info(json.dumps({
            "method": "POST", "path": "/api/delete",
            "status_code": e.status_code, "process_time": process_time
        }))
        raise
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error deleting document: {str(e)}")
        logger.info(json.dumps({
            "method": "POST", "path": "/api/delete",
            "status_code": 500, "process_time": process_time
        }))
        raise HTTPException(status_code=500, detail="文档删除失败")

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class QuestionItem(BaseModel):
    question: str
    count: Optional[int] = 0


class ClusterRequest(BaseModel):
    questions: List[QuestionItem]
    threshold: Optional[float] = None


@router.post("/knowledge-gap/semantic-cluster")
async def semantic_cluster_questions(request: ClusterRequest):
    """对未命中问题做语义聚类（向量相似度 + LLM 命名）"""
    from service.semantic_cluster import semantic_cluster

    questions = [{"question": q.question, "count": q.count} for q in request.questions]
    clusters = semantic_cluster.cluster(questions, threshold=request.threshold)
    return {"clusters": clusters}

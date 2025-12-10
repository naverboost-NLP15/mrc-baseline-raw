"""
배치 Retrieval 유틸리티
"""
from typing import List, Optional

import pandas as pd
from datasets import Dataset

from .retrieve import QdrantHybridRetriever
from src.config.retriever_config import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_N,
    DEFAULT_ALPHA,
    DEFAULT_SPARSE_TYPE,
)


def batch_retrieve(
    retriever: QdrantHybridRetriever,
    dataset: Dataset,
    topk: int = DEFAULT_TOP_K,
    topn: int = DEFAULT_TOP_N,
    alpha: float = DEFAULT_ALPHA,
    sparse_type: str = DEFAULT_SPARSE_TYPE,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    대량의 쿼리를 배치로 검색합니다.
    
    Args:
        retriever: QdrantHybridRetriever 인스턴스
        dataset: 검색할 Dataset
        topk: 최종 반환할 문서 수
        topn: Reranking 전 후보 수
        alpha: Dense 가중치
        sparse_type: Sparse 벡터 타입
        batch_size: 배치 크기
    
    Returns:
        전체 검색 결과 DataFrame
    """
    all_results = []
    
    total = len(dataset)
    for i in range(0, total, batch_size):
        batch_dataset = dataset.select(range(i, min(i + batch_size, total)))
        
        batch_df = retriever.retrieve(
            query_or_dataset=batch_dataset,
            topk=topk,
            topn=topn,
            alpha=alpha,
            sparse_type=sparse_type,
        )
        
        all_results.append(batch_df)
    
    return pd.concat(all_results, ignore_index=True)


def evaluate_retrieval(
    df: pd.DataFrame,
    topk: int = DEFAULT_TOP_K,
) -> dict:
    """
    Retrieval 결과를 평가합니다.
    
    Args:
        df: 검색 결과 DataFrame (answers 컬럼 필요)
        topk: 평가할 top-k
    
    Returns:
        평가 메트릭 딕셔너리
    """
    if "answers" not in df.columns:
        return {"error": "No 'answers' column found"}
    
    correct_count = 0
    total_count = len(df)
    
    for _, row in df.iterrows():
        answer_texts = row["answers"]["text"]
        context = row["context"]
        
        if any(ans in context for ans in answer_texts):
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        f"top{topk}_accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
    }

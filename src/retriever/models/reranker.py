"""
Reranker 모듈
"""
from typing import List, Tuple

import torch
from sentence_transformers import CrossEncoder


class Reranker:
    """CrossEncoder 기반 Reranker"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = False,
        device: str = None,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading Reranker: {model_name} (FP16={use_fp16})")
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            model_kwargs={
                "dtype": torch.float16 if use_fp16 else torch.float32
            },
        )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
    ) -> List[Tuple[int, float]]:
        """
        문서들을 쿼리와의 관련성으로 재정렬합니다.
        
        Args:
            query: 쿼리 텍스트
            documents: 문서 텍스트 리스트
            top_k: 반환할 상위 문서 수 (None이면 전체)
        
        Returns:
            (문서 인덱스, 점수) 리스트 (점수 내림차순)
        """
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        # (인덱스, 점수) 리스트 생성 및 정렬
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: int = None,
    ) -> List[List[Tuple[int, float]]]:
        """
        여러 쿼리에 대해 배치로 재정렬합니다.
        
        Args:
            queries: 쿼리 리스트
            documents_list: 각 쿼리별 문서 리스트
            top_k: 각 쿼리별 반환할 상위 문서 수
        
        Returns:
            각 쿼리별 (문서 인덱스, 점수) 리스트
        """
        results = []
        for query, documents in zip(queries, documents_list):
            result = self.rerank(query, documents, top_k)
            results.append(result)
        return results

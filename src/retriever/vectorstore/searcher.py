"""
Qdrant 검색 모듈
"""
from typing import List, Tuple

from tqdm import tqdm
from qdrant_client import models

from .qdrant_client import QdrantClientWrapper


class QdrantSearcher:
    """Qdrant 검색 수행"""
    
    def __init__(self, client_wrapper: QdrantClientWrapper, collection_name: str):
        self.client = client_wrapper.get_client()
        self.collection_name = collection_name
    
    def search_dense(
        self,
        query_vectors: List[List[float]],
        limit: int = 50,
        batch_size: int = 64,
    ) -> List[List[Tuple[int, float]]]:
        """
        Dense 벡터로 검색합니다.
        
        Returns:
            각 쿼리별 (문서 ID, 점수) 리스트
        """
        requests = [
            models.SearchRequest(
                vector=models.NamedVector(name="dense", vector=vec.tolist() if hasattr(vec, 'tolist') else vec),
                limit=limit,
                with_payload=False,
            )
            for vec in query_vectors
        ]
        
        results = []
        for i in tqdm(range(0, len(requests), batch_size), desc="Dense Search"):
            batch = requests[i : i + batch_size]
            batch_results = self.client.search_batch(
                collection_name=self.collection_name, requests=batch
            )
            results.extend(batch_results)
        
        return [
            [(hit.id, hit.score) for hit in query_results]
            for query_results in results
        ]
    
    def search_sparse(
        self,
        query_vectors: List[models.SparseVector],
        sparse_name: str = "splade",
        limit: int = 50,
        batch_size: int = 64,
    ) -> List[List[Tuple[int, float]]]:
        """
        Sparse 벡터로 검색합니다.
        
        Args:
            query_vectors: Sparse 쿼리 벡터 리스트
            sparse_name: Sparse 벡터 이름 ("splade", "bm25", "custom_sparse")
            limit: 반환할 결과 수
            batch_size: 배치 크기
        
        Returns:
            각 쿼리별 (문서 ID, 점수) 리스트
        """
        requests = [
            models.SearchRequest(
                vector=models.NamedSparseVector(name=sparse_name, vector=vec),
                limit=limit,
                with_payload=False,
            )
            for vec in query_vectors
        ]
        
        results = []
        for i in tqdm(range(0, len(requests), batch_size), desc=f"{sparse_name} Search"):
            batch = requests[i : i + batch_size]
            batch_results = self.client.search_batch(
                collection_name=self.collection_name, requests=batch
            )
            results.extend(batch_results)
        
        return [
            [(hit.id, hit.score) for hit in query_results]
            for query_results in results
        ]

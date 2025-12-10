"""
Score Fusion 모듈 (RRF 등)
"""
from typing import Dict, List, Tuple


class RRFFusion:
    """Reciprocal Rank Fusion (RRF)"""
    
    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF 파라미터 (일반적으로 60)
        """
        self.k = k
    
    def fuse(
        self,
        results_list: List[List[Tuple[int, float]]],
        weights: List[float] = None,
    ) -> List[Tuple[int, float]]:
        """
        여러 검색 결과를 RRF로 융합합니다.
        
        Args:
            results_list: 각 검색 방법별 (문서 ID, 점수) 리스트
            weights: 각 검색 방법의 가중치 (None이면 균등)
        
        Returns:
            융합된 (문서 ID, RRF 점수) 리스트 (점수 내림차순)
        """
        if weights is None:
            weights = [1.0] * len(results_list)
        
        doc_scores: Dict[int, float] = {}
        
        for results, weight in zip(results_list, weights):
            for rank, (doc_id, _) in enumerate(results):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += weight * (1.0 / (self.k + rank + 1))
        
        # 점수 내림차순 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_docs
    
    def fuse_batch(
        self,
        batch_results_list: List[List[List[Tuple[int, float]]]],
        weights: List[float] = None,
    ) -> List[List[Tuple[int, float]]]:
        """
        배치로 여러 쿼리의 검색 결과를 융합합니다.
        
        Args:
            batch_results_list: [검색방법][쿼리][결과] 형태의 3차원 리스트
            weights: 각 검색 방법의 가중치
        
        Returns:
            각 쿼리별 융합된 결과 리스트
        """
        num_queries = len(batch_results_list[0])
        fused_results = []
        
        for query_idx in range(num_queries):
            query_results = [
                method_results[query_idx] for method_results in batch_results_list
            ]
            fused = self.fuse(query_results, weights)
            fused_results.append(fused)
        
        return fused_results


def hybrid_fusion(
    dense_results: List[Tuple[int, float]],
    sparse_results: List[Tuple[int, float]],
    alpha: float = 0.5,
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Dense와 Sparse 검색 결과를 RRF로 융합합니다.
    
    Args:
        dense_results: Dense 검색 결과
        sparse_results: Sparse 검색 결과
        alpha: Dense 가중치 (1 - alpha = Sparse 가중치)
        k: RRF 파라미터
    
    Returns:
        융합된 (문서 ID, RRF 점수) 리스트
    """
    fusion = RRFFusion(k=k)
    weights = [alpha, 1.0 - alpha]
    return fusion.fuse([dense_results, sparse_results], weights)

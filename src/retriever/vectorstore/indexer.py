"""
Qdrant 인덱싱 모듈
"""
from typing import Dict, List

import torch
from tqdm import tqdm
from qdrant_client.models import PointStruct, SparseVector

from .qdrant_client import QdrantClientWrapper


class QdrantIndexer:
    """Qdrant에 벡터를 업로드하는 인덱서"""
    
    def __init__(self, client_wrapper: QdrantClientWrapper, collection_name: str):
        self.client = client_wrapper.get_client()
        self.collection_name = collection_name
    
    def upsert_points(self, points: List[PointStruct]) -> None:
        """포인트들을 업로드합니다."""
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def build_point(
        self,
        point_id: int,
        dense_vector: List[float],
        splade_indices: List[int] = None,
        splade_values: List[float] = None,
        bm25_indices: List[int] = None,
        bm25_values: List[float] = None,
        payload: Dict = None,
    ) -> PointStruct:
        """
        단일 포인트를 생성합니다.
        """
        vectors = {"dense": dense_vector}
        
        if splade_indices is not None and splade_values is not None:
            vectors["splade"] = SparseVector(
                indices=splade_indices, values=splade_values
            )
        else:
            vectors["splade"] = SparseVector(indices=[], values=[])
        
        if bm25_indices is not None and bm25_values is not None:
            vectors["bm25"] = SparseVector(indices=bm25_indices, values=bm25_values)
        else:
            vectors["bm25"] = SparseVector(indices=[], values=[])
        
        vectors["custom_sparse"] = SparseVector(indices=[], values=[])
        
        return PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload or {},
        )
    
    def index_batch(
        self,
        batch_ids: List[int],
        dense_embeddings: List[List[float]],
        splade_vectors: List[tuple] = None,
        bm25_vectors: List[tuple] = None,
        payloads: List[Dict] = None,
    ) -> None:
        """
        배치 단위로 인덱싱합니다.
        
        Args:
            batch_ids: 포인트 ID 리스트
            dense_embeddings: Dense 임베딩 리스트
            splade_vectors: (indices, values) 튜플 리스트
            bm25_vectors: (indices, values) 튜플 리스트
            payloads: 페이로드 딕셔너리 리스트
        """
        points = []
        
        for i, point_id in enumerate(batch_ids):
            dense_vec = dense_embeddings[i]
            
            splade_idx, splade_val = None, None
            if splade_vectors is not None:
                splade_idx, splade_val = splade_vectors[i]
            
            bm25_idx, bm25_val = None, None
            if bm25_vectors is not None:
                bm25_idx, bm25_val = bm25_vectors[i]
            
            payload = payloads[i] if payloads is not None else None
            
            point = self.build_point(
                point_id=point_id,
                dense_vector=dense_vec if isinstance(dense_vec, list) else dense_vec.tolist(),
                splade_indices=splade_idx,
                splade_values=splade_val,
                bm25_indices=bm25_idx,
                bm25_values=bm25_val,
                payload=payload,
            )
            points.append(point)
        
        self.upsert_points(points)
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()

"""
Qdrant 클라이언트 래퍼
"""
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
)

from src.config.qdrant_config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_HTTPS,
    QDRANT_TIMEOUT,
)


class QdrantClientWrapper:
    """Qdrant 클라이언트 래퍼"""
    
    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        api_key: str = QDRANT_API_KEY,
        https: bool = QDRANT_HTTPS,
        timeout: int = QDRANT_TIMEOUT,
    ):
        self.host = host
        self.port = port
        
        print(f"Connecting to Qdrant ({host}:{port})...")
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=https,
            timeout=timeout,
        )
    
    def collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부 확인"""
        return self.client.collection_exists(collection_name)
    
    def create_hybrid_collection(
        self,
        collection_name: str,
        dense_dim: int,
        sparse_vector_names: List[str] = None,
        recreate: bool = True,
    ) -> None:
        """
        Hybrid 컬렉션 생성 (Dense + Sparse)
        
        Args:
            collection_name: 컬렉션 이름
            dense_dim: Dense vector 차원
            sparse_vector_names: Sparse vector 이름 리스트
            recreate: 기존 컬렉션 삭제 후 재생성 여부
        """
        if sparse_vector_names is None:
            sparse_vector_names = ["splade", "bm25", "custom_sparse"]
        
        sparse_config = {
            name: SparseVectorParams() for name in sparse_vector_names
        }
        
        print(f"Creating collection '{collection_name}' with Hybrid Config...")
        
        if recreate:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config=sparse_config,
            )
        else:
            if not self.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
                    },
                    sparse_vectors_config=sparse_config,
                )
    
    def get_collection_count(self, collection_name: str) -> int:
        """컬렉션의 벡터 수 반환"""
        return self.client.count(collection_name=collection_name).count
    
    def get_client(self) -> QdrantClient:
        """원본 QdrantClient 반환"""
        return self.client

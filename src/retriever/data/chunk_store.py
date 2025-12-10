"""
청크 데이터 저장/로드 모듈
"""
import os
import pickle
from typing import Dict, List, Optional, Tuple


class ChunkStore:
    """청크 데이터 캐시 저장/로드"""
    
    def __init__(self, data_path: str, chunk_size: int, chunk_overlap: int):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_filename = f"wiki_chunks_{chunk_size}_overlap{chunk_overlap}.pkl"
        self.cache_path = os.path.join(data_path, self.cache_filename)
    
    def exists(self) -> bool:
        """캐시 파일이 존재하는지 확인"""
        return os.path.isfile(self.cache_path)
    
    def load(self) -> Optional[Dict]:
        """캐시된 청크 데이터를 로드합니다."""
        if not self.exists():
            return None
        
        print(f"Chunking Cache 로드 중... ({self.cache_path})")
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        
        return data
    
    def save(
        self,
        texts: List[str],
        titles: List[str],
        doc_ids: List[int],
        ids: List[int],
    ) -> None:
        """청크 데이터를 캐시에 저장합니다."""
        data = {
            "texts": texts,
            "titles": titles,
            "doc_ids": doc_ids,
            "ids": ids,
        }
        
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Chunking 결과 저장 완료: {self.cache_path}")
    
    def load_or_create(
        self,
        texts: List[str],
        titles: List[str],
        doc_ids: List[int],
        ids: List[int],
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        캐시가 있으면 로드하고, 없으면 저장 후 반환합니다.
        """
        cached = self.load()
        
        if cached is not None:
            return (
                cached["texts"],
                cached["titles"],
                cached["doc_ids"],
                cached["ids"],
            )
        
        self.save(texts, titles, doc_ids, ids)
        return texts, titles, doc_ids, ids

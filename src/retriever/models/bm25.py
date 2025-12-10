"""
BM25 관련 모듈
"""
import binascii
from collections import Counter
from typing import List, Tuple

from qdrant_client import models


def hash_token(token: str) -> int:
    """토큰을 unsigned 32-bit int로 해싱"""
    return binascii.crc32(token.encode("utf-8")) & 0xFFFFFFFF


def compute_bm25_weight(
    tf: int, doc_len: int, avgdl: float, idf: float, k1: float = 1.2, b: float = 0.75
) -> float:
    """
    BM25의 (IDF * TF component) 계산
    
    Args:
        tf: Term frequency
        doc_len: Document length
        avgdl: Average document length
        idf: Inverse document frequency
        k1: BM25 k1 parameter
        b: BM25 b parameter
    
    Returns:
        BM25 weight
    """
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
    return idf * (numerator / denominator)


class BM25Encoder:
    """BM25 쿼리/문서 인코더"""
    
    def __init__(self, tokenizer_fn, bm25_index=None, avgdl: float = None):
        """
        Args:
            tokenizer_fn: 토크나이저 함수 (text -> List[str])
            bm25_index: rank_bm25.BM25Okapi 인덱스 (IDF 정보용)
            avgdl: 평균 문서 길이
        """
        self.tokenizer_fn = tokenizer_fn
        self.bm25_index = bm25_index
        self.avgdl = avgdl
    
    def encode_query(self, query: str) -> models.SparseVector:
        """
        쿼리를 BM25 sparse vector로 인코딩합니다.
        Query weight = TF (일반적으로 1.0)
        """
        tokens = self.tokenizer_fn(query)
        token_counts = Counter(tokens)
        
        indices = []
        values = []
        
        for token, count in token_counts.items():
            idx = hash_token(token)
            indices.append(idx)
            values.append(float(count))
        
        return models.SparseVector(indices=indices, values=values)
    
    def encode_document(
        self, text: str, doc_len: int = None
    ) -> Tuple[List[int], List[float]]:
        """
        문서를 BM25 sparse vector로 인코딩합니다.
        IDF를 미리 곱해서 저장 (검색 시 쿼리 TF만 보내면 됨)
        """
        tokens = self.tokenizer_fn(text)
        if doc_len is None:
            doc_len = len(tokens)
        token_counts = Counter(tokens)
        
        indices = []
        values = []
        
        for token, count in token_counts.items():
            idx = hash_token(token)
            
            # IDF 조회
            idf = 0.0
            if self.bm25_index is not None:
                idf = self.bm25_index.idf.get(token, 0.0)
            
            # BM25 weight 계산
            weight = compute_bm25_weight(count, doc_len, self.avgdl or 1.0, idf)
            
            indices.append(idx)
            values.append(weight)
        
        return indices, values

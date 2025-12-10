"""
Dense Embedding 인코더 래퍼
"""
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DenseEmbedder:
    """Dense Embedding 인코더"""
    
    def __init__(
        self,
        model_name: str = "telepix/PIXIE-Spell-Preview-1.7B",
        use_fp16: bool = False,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        model_kwargs = (
            {"torch_dtype": torch.float16}
            if use_fp16 and torch.cuda.is_available()
            else {}
        )
        
        print(f"Loading Dense Encoder: {model_name} (FP16={use_fp16})")
        self.encoder = SentenceTransformer(model_name, model_kwargs=model_kwargs)
    
    @property
    def dimension(self) -> int:
        """임베딩 차원 반환"""
        sample = self.encoder.encode(["test"], normalize_embeddings=True)
        return sample.shape[1]
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = True,
        normalize: bool = True,
        prompt_name: str = None,
    ) -> np.ndarray:
        """
        텍스트들을 dense vector로 인코딩합니다.
        
        Args:
            texts: 인코딩할 텍스트 리스트
            batch_size: 배치 크기
            show_progress_bar: 진행바 표시 여부
            normalize: L2 정규화 여부
            prompt_name: 프롬프트 이름 (query 등)
        
        Returns:
            Dense embeddings (N x D)
        """
        return self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            prompt_name=prompt_name,
        )
    
    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """쿼리들을 인코딩합니다."""
        return self.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize=True,
            prompt_name="query",
        )
    
    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 8,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """문서들을 인코딩합니다."""
        return self.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize=True,
        )

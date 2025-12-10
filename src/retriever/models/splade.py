"""
SPLADE 인코더 래퍼
"""
from typing import List, Set

import numpy as np
import torch
from sentence_transformers import SparseEncoder
from transformers import AutoTokenizer
from qdrant_client import models


class SpladeEncoder:
    """SPLADE Sparse 인코더"""
    
    def __init__(
        self,
        model_name: str = "telepix/PIXIE-Splade-Preview",
        use_fp16: bool = False,
        device: str = None,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        model_kwargs = (
            {"torch_dtype": torch.float16}
            if use_fp16 and torch.cuda.is_available()
            else {}
        )
        
        print(f"Loading Sparse (SPLADE) Model: {model_name} (FP16={use_fp16})")
        self.encoder = SparseEncoder(model_name, model_kwargs=model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.special_token_ids: Set[int] = set(self.tokenizer.all_special_ids)
        
        if torch.cuda.is_available():
            self.encoder.to(self.device)
    
    def _filter_special_tokens(self, embeddings: np.ndarray) -> np.ndarray:
        """Special token 필터링"""
        for sp_id in self.special_token_ids:
            if sp_id < embeddings.shape[-1]:
                embeddings[..., sp_id] = 0
        return embeddings
    
    def _to_numpy(self, output) -> np.ndarray:
        """출력을 numpy 배열로 변환"""
        if isinstance(output, torch.Tensor):
            if output.is_sparse:
                output = output.to_dense()
            output = output.cpu().numpy()
        
        if output.dtype == np.float16:
            output = output.astype(np.float32)
        
        return output
    
    def encode_queries(
        self, queries: List[str], batch_size: int = 16
    ) -> List[models.SparseVector]:
        """
        쿼리들을 SPLADE sparse vector로 인코딩합니다.
        """
        sparse_vectors = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    embeddings = self.encoder.encode_query(batch)
            
            embeddings = self._to_numpy(embeddings)
            embeddings = self._filter_special_tokens(embeddings)
            
            for row in embeddings:
                indices = np.nonzero(row)[0].tolist()
                values = row[indices].tolist()
                sparse_vectors.append(
                    models.SparseVector(indices=indices, values=values)
                )
        
        return sparse_vectors
    
    def encode_documents(
        self, documents: List[str], batch_size: int = 8
    ) -> List[models.SparseVector]:
        """
        문서들을 SPLADE sparse vector로 인코딩합니다.
        """
        sparse_vectors = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    embeddings = self.encoder.encode_document(batch)
            
            embeddings = self._to_numpy(embeddings)
            embeddings = self._filter_special_tokens(embeddings)
            
            for row in embeddings:
                indices = np.nonzero(row)[0].tolist()
                values = row[indices].tolist()
                sparse_vectors.append(
                    models.SparseVector(indices=indices, values=values)
                )
        
        return sparse_vectors

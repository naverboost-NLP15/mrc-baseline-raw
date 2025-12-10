"""
Qdrant Hybrid Retriever 파이프라인
E2E: question → search → rank → result
"""
import json
import os
import pickle
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from src.config.qdrant_config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_COLLECTION_NAME,
)
from src.config.retriever_config import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_N,
    DEFAULT_ALPHA,
    DEFAULT_SPARSE_TYPE,
    USE_RERANKER,
    USE_FP16,
    RRF_K,
)
from src.retriever.data.chunk_store import ChunkStore
from src.retriever.data.preprocessor import TextPreprocessor
from src.retriever.models.dense_embedder import DenseEmbedder
from src.retriever.models.splade import SpladeEncoder
from src.retriever.models.bm25 import BM25Encoder
from src.retriever.models.reranker import Reranker
from src.retriever.vectorstore.qdrant_client import QdrantClientWrapper
from src.retriever.vectorstore.searcher import QdrantSearcher
from src.retriever.vectorstore.fusion import hybrid_fusion


class QdrantHybridRetriever:
    """Qdrant 기반 Hybrid Retriever (Dense + Sparse + Reranker)"""
    
    def __init__(
        self,
        data_path: str = "raw/data",
        context_path: str = "wikipedia_documents.json",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        dense_model_name: str = DEFAULT_DENSE_MODEL,
        sparse_model_name: str = DEFAULT_SPARSE_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
        use_reranker: bool = USE_RERANKER,
        use_fp16: bool = USE_FP16,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        qdrant_api_key: str = QDRANT_API_KEY,
    ):
        self.data_path = data_path
        self.collection_name = collection_name
        self.use_reranker = use_reranker
        
        # 1. 청크 데이터 로드 (Reranking 및 텍스트 반환용)
        chunk_store = ChunkStore(data_path, chunk_size, chunk_overlap)
        cached = chunk_store.load()
        
        if cached is not None:
            self.texts = cached["texts"]
            self.titles = cached["titles"]
            self.doc_ids = cached["doc_ids"]
            self.ids = cached["ids"]
            print(f"Loaded {len(self.texts)} cached chunks")
        else:
            raise ValueError(
                f"Chunk cache not found. Please run build_vector_db.py first."
            )
        
        # 2. 전처리기 (토크나이징용)
        self.preprocessor = TextPreprocessor(chunk_size, chunk_overlap)
        
        # 3. 모델 로드
        self.dense_encoder = DenseEmbedder(dense_model_name, use_fp16=use_fp16)
        self.splade_encoder = SpladeEncoder(sparse_model_name, use_fp16=use_fp16)
        self.bm25_encoder = BM25Encoder(self.preprocessor.kiwi_tokenize)
        
        # 4. Qdrant 연결
        self.client_wrapper = QdrantClientWrapper(
            host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key
        )
        
        if not self.client_wrapper.collection_exists(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' does not exist in Qdrant!"
            )
        
        self.searcher = QdrantSearcher(self.client_wrapper, collection_name)
        
        # 5. Reranker
        self.reranker = None
        if use_reranker:
            self.reranker = Reranker(reranker_model_name, use_fp16=use_fp16)
    
    def retrieve(
        self,
        query_or_dataset: Union[str, List[str], Dataset],
        topk: int = DEFAULT_TOP_K,
        topn: int = DEFAULT_TOP_N,
        alpha: float = DEFAULT_ALPHA,
        sparse_type: str = DEFAULT_SPARSE_TYPE,
    ) -> pd.DataFrame:
        """
        Hybrid Search (Dense + Sparse) → Reranking
        
        Args:
            query_or_dataset: 쿼리 문자열, 쿼리 리스트, 또는 Dataset
            topk: 최종 반환할 문서 수
            topn: Reranking 전 후보 수
            alpha: Dense 가중치 (0.0 = Sparse Only, 1.0 = Dense Only)
            sparse_type: Sparse 벡터 타입 ("splade", "bm25")
        
        Returns:
            검색 결과 DataFrame
        """
        # 쿼리 준비
        if isinstance(query_or_dataset, str):
            queries = [query_or_dataset]
            dataset = None
        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]
            dataset = query_or_dataset
        else:
            queries = query_or_dataset
            dataset = None
        
        # 후보 수 조정
        topn_candidate = min(len(self.ids), max(topk * 5, topn))
        
        # 1. 쿼리 인코딩
        print(f"Encoding queries... (Dense + {sparse_type})")
        
        dense_query_embs = self.dense_encoder.encode_queries(queries)
        
        if sparse_type == "splade":
            sparse_query_vecs = self.splade_encoder.encode_queries(queries)
        elif sparse_type == "bm25":
            sparse_query_vecs = [
                self.bm25_encoder.encode_query(q) for q in tqdm(queries, desc="BM25 Encoding")
            ]
        else:
            raise ValueError(f"Unknown sparse_type: {sparse_type}")
        
        # 2. Qdrant 검색
        print(f"Searching Qdrant (Top-{topn_candidate})...")
        
        dense_results = self.searcher.search_dense(dense_query_embs, limit=topn_candidate)
        sparse_results = self.searcher.search_sparse(
            sparse_query_vecs, sparse_name=sparse_type, limit=topn_candidate
        )
        
        # 3. RRF Fusion
        print("Fusing results (RRF)...")
        
        total_results = []
        
        for i, query in enumerate(tqdm(queries, desc="Processing")):
            # Fusion
            fused = hybrid_fusion(
                dense_results[i], sparse_results[i], alpha=alpha, k=RRF_K
            )
            
            rerank_count = min(len(fused), topn)
            candidates = fused[:rerank_count]
            
            # 4. Reranking
            if self.use_reranker and self.reranker is not None:
                candidate_indices = [c[0] for c in candidates]
                candidate_texts = [
                    f"{self.titles[idx]} {self.texts[idx]}"
                    for idx in candidate_indices
                ]
                
                reranked = self.reranker.rerank(query, candidate_texts, top_k=topk)
                final_indices = [candidate_indices[r[0]] for r in reranked]
            else:
                final_indices = [c[0] for c in candidates[:topk]]
            
            # 결과 구성
            context = "\n\n".join([self.texts[idx] for idx in final_indices])
            retrieved_doc_ids = [self.doc_ids[idx] for idx in final_indices]
            
            result = {
                "question": query,
                "id": dataset[i]["id"] if dataset is not None else str(i),
                "context": context,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_indices": final_indices,  # JSON 저장용 인덱스
            }
            
            # 추가 정보 (dataset에 있는 경우)
            if dataset is not None:
                if "answers" in dataset.features:
                    result["answers"] = dataset[i]["answers"]
                if "context" in dataset.features:
                    result["original_context"] = dataset[i]["context"]
                if "document_id" in dataset.features:
                    result["original_document_id"] = dataset[i]["document_id"]
            
            total_results.append(result)
        
        return pd.DataFrame(total_results)
    
    def save_to_json(
        self,
        df: pd.DataFrame,
        output_path: str,
        topk: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
        sparse_type: str = DEFAULT_SPARSE_TYPE,
    ) -> str:
        """
        검색 결과를 JSON 파일로 저장합니다.
        
        JSON 형식:
        [
            {
                "question": "질문",
                "id": "mrc-1-000653",
                "contexts": [
                    {"text": "문서 텍스트", "doc_id": 24024, "score": 188.48}
                ]
            }
        ]
        
        Args:
            df: 검색 결과 DataFrame
            output_path: 저장할 JSON 파일 경로
            topk: 검색에 사용된 topk
            alpha: Dense 가중치
            sparse_type: Sparse 타입
        
        Returns:
            저장된 파일 경로
        """
        # DataFrame을 JSON 형식으로 변환
        data = []
        for _, row in df.iterrows():
            # contexts 생성 (개별 문서 정보 포함)
            contexts = []
            for idx in row["retrieved_indices"]:
                contexts.append({
                    "text": self.texts[idx],
                    "doc_id": self.doc_ids[idx],
                    "score": 0.0,  # score는 fusion 후 정보가 없으므로 0으로 설정
                })
            
            item = {
                "question": row["question"],
                "id": row["id"],
                "contexts": contexts,
            }
            data.append(item)
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # JSON 저장 (단순 배열 형태)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved retrieval results to {output_path}")
        return output_path


if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk, concatenate_datasets
    
    parser = argparse.ArgumentParser(description="Qdrant Hybrid Retrieval")
    parser.add_argument("--dataset_name", type=str, default="raw/data/train_dataset")
    parser.add_argument("--data_path", type=str, default="raw/data")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--topn", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--sparse_type", type=str, default=DEFAULT_SPARSE_TYPE)
    parser.add_argument("--collection_name", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--no_reranker", dest="use_reranker", action="store_false")
    parser.add_argument("--fp16", action="store_true")
    parser.set_defaults(use_reranker=True)
    
    args = parser.parse_args()
    
    # Load Dataset
    print(f"Loading dataset from {args.dataset_name}...")
    original_dataset = load_from_disk(args.dataset_name)
    
    try:
        full_ds = concatenate_datasets([
            original_dataset["train"].flatten_indices(),
            original_dataset["validation"].flatten_indices(),
        ])
    except KeyError:
        full_ds = original_dataset["validation"]
    
    # Initialize Retriever
    retriever = QdrantHybridRetriever(
        data_path=args.data_path,
        collection_name=args.collection_name,
        use_reranker=args.use_reranker,
        use_fp16=args.fp16,
    )
    
    # Retrieve
    df = retriever.retrieve(
        query_or_dataset=full_ds,
        topk=args.topk,
        topn=args.topn,
        alpha=args.alpha,
        sparse_type=args.sparse_type,
    )
    
    # Evaluation
    if "answers" in df.columns:
        correct_count = 0
        for idx, row in df.iterrows():
            answer_texts = row["answers"]["text"]
            if any(ans in row["context"] for ans in answer_texts):
                correct_count += 1
        acc = correct_count / len(df)
        print(f"Top-{args.topk} Accuracy: {acc:.4f}")

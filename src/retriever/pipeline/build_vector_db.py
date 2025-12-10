"""
Qdrant Hybrid Index 구축 파이프라인
E2E: load → chunk → embed → index
"""
import os
import pickle
from collections import Counter
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix
from rank_bm25 import BM25Okapi

from src.config.qdrant_config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    DEFAULT_COLLECTION_NAME,
)
from src.retriever.data.loader import WikipediaLoader
from src.retriever.data.preprocessor import TextPreprocessor
from src.retriever.data.chunk_store import ChunkStore
from src.retriever.models.dense_embedder import DenseEmbedder
from src.retriever.models.splade import SpladeEncoder
from src.retriever.models.bm25 import hash_token, compute_bm25_weight
from src.retriever.vectorstore.qdrant_client import QdrantClientWrapper
from src.retriever.vectorstore.indexer import QdrantIndexer


def build_hybrid_index(
    data_path: str = "raw/data",
    context_path: str = "wikipedia_documents.json",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    dense_model_name: str = DEFAULT_DENSE_MODEL,
    sparse_model_name: str = DEFAULT_SPARSE_MODEL,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    qdrant_host: str = QDRANT_HOST,
    qdrant_port: int = QDRANT_PORT,
    api_key: str = QDRANT_API_KEY,
    batch_size: int = 8,
) -> None:
    """
    Qdrant Hybrid Index를 구축합니다 (Dense + SPLADE + BM25)
    
    Args:
        data_path: 데이터 경로
        context_path: 문서 파일명
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        dense_model_name: Dense 임베딩 모델
        sparse_model_name: SPLADE 모델
        collection_name: Qdrant 컬렉션 이름
        qdrant_host: Qdrant 호스트
        qdrant_port: Qdrant 포트
        api_key: Qdrant API 키
        batch_size: 배치 크기
    """
    # 1. 데이터 로드 및 청킹
    print("=== Step 1: Loading and Chunking Documents ===")
    
    loader = WikipediaLoader(data_path, context_path)
    preprocessor = TextPreprocessor(chunk_size, chunk_overlap)
    chunk_store = ChunkStore(data_path, chunk_size, chunk_overlap)
    
    # 캐시 확인
    cached = chunk_store.load()
    if cached is not None:
        texts = cached["texts"]
        titles = cached["titles"]
        doc_ids = cached["doc_ids"]
        ids = cached["ids"]
        print(f"Loaded {len(texts)} chunks from cache")
    else:
        raw_texts, raw_titles, raw_doc_ids = loader.load_as_lists()
        texts, titles, doc_ids, ids = preprocessor.chunk_documents(
            raw_texts, raw_titles, raw_doc_ids
        )
        chunk_store.save(texts, titles, doc_ids, ids)
        print(f"Created and cached {len(texts)} chunks")
    
    # 2. BM25 인덱스 로드/생성
    print("=== Step 2: Building BM25 Index ===")
    
    bm25_path = os.path.join(
        data_path, f"bm25_wiki_chunk{chunk_size}_overlap{chunk_overlap}_v3.pkl"
    )
    
    if os.path.isfile(bm25_path):
        print(f"Loading existing BM25 index from {bm25_path}")
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
    else:
        print("Building new BM25 index...")
        tokenized_corpus = [
            preprocessor.kiwi_tokenize(f"{title} {text}")
            for title, text in tqdm(zip(titles, texts), total=len(texts), desc="Tokenizing")
        ]
        bm25 = BM25Okapi(corpus=tokenized_corpus)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        print("BM25 index built and saved")
    
    avgdl = bm25.avgdl
    print(f"BM25 Average Document Length: {avgdl}")
    
    # 3. 모델 로드
    print("=== Step 3: Loading Embedding Models ===")
    
    dense_encoder = DenseEmbedder(dense_model_name)
    splade_encoder = SpladeEncoder(sparse_model_name)
    
    dense_dim = dense_encoder.dimension
    print(f"Dense Vector dimension: {dense_dim}")
    
    # 4. Qdrant 컬렉션 생성
    print("=== Step 4: Creating Qdrant Collection ===")
    
    client_wrapper = QdrantClientWrapper(
        host=qdrant_host, port=qdrant_port, api_key=api_key
    )
    client_wrapper.create_hybrid_collection(
        collection_name=collection_name,
        dense_dim=dense_dim,
        recreate=True,
    )
    
    indexer = QdrantIndexer(client_wrapper, collection_name)
    
    # 5. 배치 인코딩 및 업로드
    print("=== Step 5: Indexing Documents ===")
    
    total_docs = len(texts)
    
    for i in tqdm(range(0, total_docs, batch_size), desc="Indexing to Qdrant"):
        batch_texts = texts[i : i + batch_size]
        batch_titles = titles[i : i + batch_size]
        batch_doc_ids = doc_ids[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        
        # 임베딩용 텍스트 준비
        docs_for_embed = [f"{t}\n{txt}" for t, txt in zip(batch_titles, batch_texts)]
        
        # Dense 인코딩
        dense_embeddings = dense_encoder.encode_documents(
            docs_for_embed, batch_size=batch_size, show_progress_bar=False
        )
        
        # SPLADE 인코딩
        splade_vectors = splade_encoder.encode_documents(docs_for_embed, batch_size=batch_size)
        splade_tuples = [
            (vec.indices, vec.values) for vec in splade_vectors
        ]
        
        # BM25 벡터 생성
        bm25_tuples = []
        for doc_text in docs_for_embed:
            tokens = preprocessor.kiwi_tokenize(doc_text)
            doc_len = len(tokens)
            token_counts = Counter(tokens)
            
            indices = []
            values = []
            for token, count in token_counts.items():
                idx = hash_token(token)
                idf = bm25.idf.get(token, 0.0)
                weight = compute_bm25_weight(count, doc_len, avgdl, idf)
                indices.append(idx)
                values.append(weight)
            
            bm25_tuples.append((indices, values))
        
        # 페이로드 생성
        payloads = [
            {
                "title": batch_titles[j],
                "text": batch_texts[j],
                "doc_id": batch_doc_ids[j],
                "chunk_id": batch_ids[j],
            }
            for j in range(len(batch_texts))
        ]
        
        # 인덱싱
        indexer.index_batch(
            batch_ids=batch_ids,
            dense_embeddings=dense_embeddings,
            splade_vectors=splade_tuples,
            bm25_vectors=bm25_tuples,
            payloads=payloads,
        )
    
    print("=== Indexing Complete! ===")
    print(f"Total vectors in collection '{collection_name}': {client_wrapper.get_collection_count(collection_name)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Qdrant Hybrid Index")
    parser.add_argument("--data_path", type=str, default="raw/data")
    parser.add_argument("--context_path", type=str, default="wikipedia_documents.json")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--dense_model", type=str, default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--sparse_model", type=str, default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--collection_name", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--api_key", type=str, default=QDRANT_API_KEY)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    build_hybrid_index(
        data_path=args.data_path,
        context_path=args.context_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dense_model_name=args.dense_model,
        sparse_model_name=args.sparse_model,
        collection_name=args.collection_name,
        api_key=args.api_key,
        batch_size=args.batch_size,
    )

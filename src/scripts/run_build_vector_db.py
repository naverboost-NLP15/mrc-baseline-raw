#!/usr/bin/env python
"""
Qdrant Hybrid Vector DB 구축 스크립트

Usage:
    python -m src.scripts.run_build_vector_db \\
        --data_path raw/data \\
        --collection_name hybird_collection_v1 \\
        --chunk_size 1000 \\
        --chunk_overlap 200
"""
import argparse

from src.retriever.pipeline.build_vector_db import build_hybrid_index
from src.config.qdrant_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    DEFAULT_COLLECTION_NAME,
    QDRANT_API_KEY,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build Qdrant Hybrid Index (Dense + SPLADE + BM25)"
    )
    parser.add_argument(
        "--data_path", type=str, default="raw/data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--context_path", type=str, default="wikipedia_documents.json",
        help="Context file name"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap for text splitting"
    )
    parser.add_argument(
        "--dense_model", type=str, default=DEFAULT_DENSE_MODEL,
        help="Dense embedding model name"
    )
    parser.add_argument(
        "--sparse_model", type=str, default=DEFAULT_SPARSE_MODEL,
        help="Sparse (SPLADE) embedding model name"
    )
    parser.add_argument(
        "--collection_name", type=str, default=DEFAULT_COLLECTION_NAME,
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--api_key", type=str, default=QDRANT_API_KEY,
        help="Qdrant API Key"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for encoding"
    )
    
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


if __name__ == "__main__":
    main()

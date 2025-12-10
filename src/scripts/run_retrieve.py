#!/usr/bin/env python
"""
Retrieval 단독 실행 스크립트
결과를 JSON 파일로 저장하여 Reader와 독립적으로 실행 가능

Usage:
    python -m src.scripts.run_retrieve \\
        --dataset_name raw/data/train_dataset \\
        --topk 5 \\
        --alpha 0.5 \\
        --output_json output/retrieval/retrieval_results.json
"""
import argparse
import os

import wandb
from datasets import load_from_disk, concatenate_datasets

from src.retriever.pipeline.retrieve import QdrantHybridRetriever
from src.retriever.utils.timer import timer
from src.config.retriever_config import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_N,
    DEFAULT_ALPHA,
    DEFAULT_SPARSE_TYPE,
)
from src.config.qdrant_config import DEFAULT_COLLECTION_NAME


def main():
    parser = argparse.ArgumentParser(description="Qdrant Hybrid Retrieval")
    parser.add_argument(
        "--dataset_name", type=str, default="raw/data/train_dataset",
        help="Dataset path"
    )
    parser.add_argument(
        "--data_path", type=str, default="raw/data",
        help="Data directory path"
    )
    parser.add_argument(
        "--topk", type=int, default=DEFAULT_TOP_K,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--topn", type=int, default=DEFAULT_TOP_N,
        help="Number of candidates before reranking"
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="Dense weight (0.0~1.0)"
    )
    parser.add_argument(
        "--sparse_type", type=str, default=DEFAULT_SPARSE_TYPE,
        choices=["splade", "bm25"],
        help="Sparse vector type"
    )
    parser.add_argument(
        "--collection_name", type=str, default=DEFAULT_COLLECTION_NAME,
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--no_reranker", dest="use_reranker", action="store_false",
        help="Disable reranker"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use FP16 for models"
    )
    parser.add_argument(
        "--output_json", type=str, default="output/retrieval/retrieval_results.json",
        help="Output JSON file path for retrieval results"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="QDQA_Retrieval",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_name", type=str, default=None,
        help="WandB run name"
    )
    parser.set_defaults(use_reranker=True)
    
    args = parser.parse_args()
    
    # WandB 초기화
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )
    
    # 데이터셋 로드
    print(f"Loading dataset from {args.dataset_name}...")
    original_dataset = load_from_disk(args.dataset_name)
    
    try:
        full_ds = concatenate_datasets([
            original_dataset["train"].flatten_indices(),
            original_dataset["validation"].flatten_indices(),
        ])
    except KeyError:
        full_ds = original_dataset["validation"]
    
    # Retriever 초기화
    retriever = QdrantHybridRetriever(
        data_path=args.data_path,
        collection_name=args.collection_name,
        use_reranker=args.use_reranker,
        use_fp16=args.fp16,
    )
    
    # Retrieve
    with timer("Hybrid Search (Qdrant)"):
        df = retriever.retrieve(
            query_or_dataset=full_ds,
            topk=args.topk,
            topn=args.topn,
            alpha=args.alpha,
            sparse_type=args.sparse_type,
        )
    
    # 평가
    if "answers" in df.columns:
        correct_count = 0
        for _, row in df.iterrows():
            answer_texts = row["answers"]["text"]
            if any(ans in row["context"] for ans in answer_texts):
                correct_count += 1
        acc = correct_count / len(df)
        print(f"Top-{args.topk} Accuracy: {acc:.4f}")
        
        wandb.log({
            f"top{args.topk}_accuracy": acc,
            "correct_count": correct_count,
            "total_count": len(df),
        })
    
    # JSON 파일로 저장
    output_json = retriever.save_to_json(
        df=df,
        output_path=args.output_json,
        topk=args.topk,
        alpha=args.alpha,
        sparse_type=args.sparse_type,
    )
    
    print(f"\n{'='*60}")
    print(f"Retrieval complete!")
    print(f"Results saved to: {output_json}")
    print(f"Use this JSON file with the Reader:")
    print(f"  python -m src.scripts.run_reader_inference \\")
    print(f"      --retrieval_json_path {output_json} \\")
    print(f"      --model_name_or_path <your_model_path>")
    print(f"{'='*60}")
    
    wandb.finish()


if __name__ == "__main__":
    main()

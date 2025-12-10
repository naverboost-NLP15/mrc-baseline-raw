#!/usr/bin/env python
"""
전체 ODQA 파이프라인 실행 스크립트 (Retrieve → JSON → Read)

Retriever와 Reader가 JSON 파일을 통해 독립적으로 통신합니다.

Usage:
    python -m src.scripts.run_all_pipeline \\
        --model_name_or_path output/models/my_model \\
        --dataset_name raw/data/test_dataset \\
        --output_dir output/predictions/full_pipeline \\
        --topk 5 \\
        --alpha 0.5
"""
import argparse
import os

from datasets import load_from_disk
from transformers import TrainingArguments

from src.utils.arguments import ModelArguments, DataTrainingArguments
from src.retriever.pipeline.retrieve import QdrantHybridRetriever
from src.reader.inference import run_inference
from src.retriever.utils.timer import timer


def main():
    parser = argparse.ArgumentParser(description="Full ODQA Pipeline (Retrieve → Read)")
    
    # Retriever arguments
    parser.add_argument("--data_path", type=str, default="raw/data")
    parser.add_argument("--dataset_name", type=str, default="raw/data/test_dataset")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sparse_type", type=str, default="splade")
    parser.add_argument("--collection_name", type=str, default="hybird_collection_v1")
    parser.add_argument("--fp16", action="store_true")
    
    # Reader arguments
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_answer_length", type=int, default=30)
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 데이터셋 로드
    print(f"Loading dataset from {args.dataset_name}...")
    original_dataset = load_from_disk(args.dataset_name)
    
    if "validation" in original_dataset:
        dataset = original_dataset["validation"]
    elif "test" in original_dataset:
        dataset = original_dataset["test"]
    else:
        dataset = original_dataset
    
    # 2. Retrieval
    print("\n=== Step 1: Retrieval ===")
    
    retriever = QdrantHybridRetriever(
        data_path=args.data_path,
        collection_name=args.collection_name,
        use_fp16=args.fp16,
    )
    
    with timer("Hybrid Retrieval"):
        df = retriever.retrieve(
            query_or_dataset=dataset,
            topk=args.topk,
            topn=args.topn,
            alpha=args.alpha,
            sparse_type=args.sparse_type,
        )
    
    # Retrieval 결과를 JSON으로 저장
    retrieval_json_path = os.path.join(args.output_dir, "retrieval_results.json")
    retriever.save_to_json(
        df=df,
        output_path=retrieval_json_path,
        topk=args.topk,
        alpha=args.alpha,
        sparse_type=args.sparse_type,
    )
    print(f"Retrieval results saved to {retrieval_json_path}")
    
    # 3. Reader 추론 (JSON 파일 기반)
    print("\n=== Step 2: Reader Inference ===")
    
    # HfArgumentParser 형식으로 변환
    model_args = ModelArguments(model_name_or_path=args.model_name_or_path)
    data_args = DataTrainingArguments(
        retrieval_json_path=retrieval_json_path,  # JSON 파일 경로 전달
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_answer_length=args.max_answer_length,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_predict="answers" not in df.columns,
        do_eval="answers" in df.columns,
        per_device_eval_batch_size=32,
        fp16=args.fp16,
    )
    
    run_inference(model_args, data_args, training_args)
    
    print("\n=== Pipeline Complete! ===")
    print(f"Results saved to {args.output_dir}")
    print(f"Retrieval JSON: {retrieval_json_path}")


if __name__ == "__main__":
    main()

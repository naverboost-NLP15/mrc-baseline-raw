#!/usr/bin/env python
"""
Reader 추론 스크립트 (JSON 파일 기반)

Retriever가 생성한 JSON 파일을 입력으로 받아 Reader 추론을 수행합니다.
Retriever와 독립적으로 실행됩니다.

Usage (JSON 파일 사용):
    python -m src.scripts.run_reader_predict \\
        --output_dir output/predictions/my_pred \\
        --model_name_or_path output/models/my_model \\
        --retrieval_json_path output/retrieval/retrieval_results.json \\
        --do_predict

Usage (기존 Dataset 사용 - context가 이미 있는 경우):
    python -m src.scripts.run_reader_predict \\
        --output_dir output/predictions/my_pred \\
        --model_name_or_path output/models/my_model \\
        --dataset_name raw/data/test_dataset \\
        --do_predict
"""
from transformers import HfArgumentParser, TrainingArguments

from src.utils.arguments import ModelArguments, DataTrainingArguments
from src.reader.inference import run_inference


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    run_inference(model_args, data_args, training_args)


if __name__ == "__main__":
    main()

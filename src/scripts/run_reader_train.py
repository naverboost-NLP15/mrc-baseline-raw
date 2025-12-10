#!/usr/bin/env python
"""
Reader 학습 스크립트

Usage:
    python -m src.scripts.run_reader_train \\
        --output_dir output/models/my_model \\
        --model_name_or_path klue/roberta-large \\
        --dataset_name raw/data/train_dataset \\
        --do_train \\
        --do_eval \\
        --per_device_train_batch_size 4 \\
        --num_train_epochs 3
"""
from transformers import HfArgumentParser, TrainingArguments

from src.utils.arguments import ModelArguments, DataTrainingArguments
from src.reader.train_qa import train_reader


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    train_reader(model_args, data_args, training_args)


if __name__ == "__main__":
    main()

"""
Reader 학습 모듈
"""
import logging
import os
import sys
from typing import NoReturn

import evaluate
from datasets import DatasetDict, load_from_disk, load_dataset, concatenate_datasets
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src.utils.arguments import ModelArguments, DataTrainingArguments
from src.utils.seed import set_seed as set_all_seeds
from src.reader.model.reader import MRCReader
from src.reader.model.trainer_qa import QuestionAnsweringTrainer
from src.reader.data_processor import DataProcessor
from src.reader.utils.postprocess import postprocess_qa_predictions
from src.reader.utils.metrics import check_no_error, compute_metrics_with_prefix

logger = logging.getLogger(__name__)


def train_reader(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
) -> NoReturn:
    """
    Reader 모델을 학습합니다.
    
    Args:
        model_args: 모델 관련 인자
        data_args: 데이터 관련 인자
        training_args: 학습 관련 인자
    """
    # Logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    logger.info("Training/evaluation parameters %s", training_args)
    
    # Seed 설정
    set_seed(training_args.seed)
    set_all_seeds(training_args.seed)
    
    # WandB 설정
    training_args.report_to = ["wandb"]
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "QDQA"
    if training_args.run_name is None:
        training_args.run_name = training_args.output_dir.split("/")[-1]
    
    # 데이터셋 로드
    print(f"Loading dataset from {data_args.dataset_name}...")
    datasets = load_from_disk(data_args.dataset_name)
    
    # KorQuad 추가 (옵션)
    if data_args.add_korquad:
        print("Adding KorQuad v1 dataset...")
        korquad = load_dataset("squad_kor_v1")
        kq_train = korquad["train"]
        
        if "train" in datasets:
            org_train = datasets["train"]
            common_columns = [
                col for col in org_train.column_names if col in kq_train.column_names
            ]
            kq_train = kq_train.select_columns(common_columns)
            org_train_filtered = org_train.select_columns(common_columns)
            kq_train = kq_train.cast(org_train_filtered.features)
            combined_train = concatenate_datasets([org_train_filtered, kq_train])
            datasets["train"] = combined_train
            print(f"Added KorQuad. New train size: {len(datasets['train'])}")
    
    # 모델 로드
    print(f"Loading model from {model_args.model_name_or_path}...")
    reader = MRCReader(
        model_name_or_path=model_args.model_name_or_path,
        config_name=model_args.config_name,
        tokenizer_name=model_args.tokenizer_name,
    )
    
    model = reader.get_model()
    tokenizer = reader.get_tokenizer()
    
    # 에러 체크
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    
    # 데이터 전처리기
    data_processor = DataProcessor(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=data_args.doc_stride,
        pad_to_max_length=data_args.pad_to_max_length,
    )
    
    # 컬럼 이름
    column_names = datasets["train"].column_names if training_args.do_train else datasets["validation"].column_names
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    
    # 학습 데이터 전처리
    train_dataset = None
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"].map(
            data_processor.prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    
    # 평가 데이터 전처리
    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = datasets["validation"].map(
            data_processor.prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=datasets["validation"].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    
    # Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    
    # Post-processing 함수
    def post_processing_function(examples, features, predictions, training_args):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        
        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )
    
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics_with_prefix,
    )
    
    # 학습
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path) and not training_args.overwrite_output_dir:
            checkpoint = model_args.model_name_or_path
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # 평가
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    train_reader(model_args, data_args, training_args)


if __name__ == "__main__":
    main()

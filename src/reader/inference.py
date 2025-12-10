"""
Reader 추론 모듈 (JSON 파일 기반 독립 실행)
Retriever와의 의존성 없이 JSON 파일을 입력으로 받아 Reader만 실행
"""
import json
import logging
import os
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src.utils.arguments import ModelArguments, DataTrainingArguments
from src.reader.model.reader import MRCReader
from src.reader.model.trainer_qa import QuestionAnsweringTrainer
from src.reader.data_processor import DataProcessor
from src.reader.utils.postprocess import postprocess_qa_predictions
from src.reader.utils.metrics import check_no_error

logger = logging.getLogger(__name__)


def load_retrieval_json(json_path: str) -> DatasetDict:
    """
    Retriever가 생성한 JSON 파일을 로드하여 DatasetDict로 변환합니다.
    contexts 배열의 모든 text를 \n\n으로 합쳐서 하나의 context로 만듭니다.
    
    Args:
        json_path: Retriever가 생성한 JSON 파일 경로
    
    Returns:
        DatasetDict with 'validation' split
    
    JSON 포맷:
        [
            {
                "question": "질문",
                "id": "mrc-1-000653",
                "contexts": [
                    {"text": "문서 텍스트", "doc_id": 24024, "score": 188.48}
                ]
            }
        ]
    """
    print(f"Loading retrieval results from {json_path}...")
    
    with open(json_path, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)
    
    print(f"Loaded {len(retrieval_data)} samples from JSON")
    
    # contexts를 합쳐서 하나의 context로 변환
    data_list = []
    for item in retrieval_data:
        # 모든 context의 text를 \n\n으로 합치기
        context_texts = [ctx["text"] for ctx in item["contexts"]]
        merged_context = "\n\n".join(context_texts)
        
        data_list.append({
            "id": item["id"],
            "question": item["question"],
            "context": merged_context,
        })
    
    # Features 정의 (answers 없음 - inference용)
    f = Features({
        "context": Value(dtype="string", id=None),
        "id": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
    })
    
    # Dataset 생성
    dataset = Dataset.from_list(data_list)
    datasets = DatasetDict({"validation": dataset.cast(f)})
    
    return datasets


def run_inference(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
) -> NoReturn:
    """
    JSON 파일 기반 Reader 추론을 수행합니다.
    
    Retriever와 독립적으로 실행되며, retrieval_json_path 또는 dataset_name을 통해
    데이터를 로드합니다.
    
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
    
    logger.info("Inference parameters %s", training_args)
    
    # Seed 설정
    set_seed(training_args.seed)
    
    # WandB 설정
    training_args.report_to = ["wandb"]
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "QDQA"
    if training_args.run_name is None:
        training_args.run_name = training_args.output_dir.split("/")[-1]
    
    # 데이터셋 로드 (JSON 파일 또는 기존 HuggingFace Dataset)
    if hasattr(data_args, "retrieval_json_path") and data_args.retrieval_json_path:
        # JSON 파일에서 로드 (Retriever 결과 파일)
        print(f"Loading retrieval results from JSON: {data_args.retrieval_json_path}")
        datasets = load_retrieval_json(data_args.retrieval_json_path)
    else:
        # 기존 HuggingFace Dataset에서 로드
        print(f"Loading dataset from {data_args.dataset_name}...")
        datasets = load_from_disk(data_args.dataset_name)
    
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
    
    column_names = datasets["validation"].column_names
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    
    # 평가 데이터 전처리
    eval_dataset = datasets["validation"].map(
        data_processor.prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    
    # Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    
    # 출력 디렉토리 설정
    if training_args.do_eval:
        training_args.output_dir = os.path.join(training_args.output_dir, "eval_pred")
    elif training_args.do_predict:
        training_args.output_dir = os.path.join(training_args.output_dir, "test")
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    
    # Post-processing 함수
    def post_processing_function(
        examples, features, predictions: Tuple[np.ndarray, np.ndarray], training_args
    ) -> EvalPrediction:
        predictions_dict = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions_dict.items()
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
    
    metric = evaluate.load("squad")
    
    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    # Trainer 초기화
    print("Initializing trainer...")
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    
    logger.info("*** Inference ***")
    
    # Predict
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        print("Prediction complete. Results saved to output directory.")
    
    # Evaluate
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    run_inference(model_args, data_args, training_args)


if __name__ == "__main__":
    main()

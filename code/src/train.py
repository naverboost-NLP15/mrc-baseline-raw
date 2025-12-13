import logging
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict
from trainer_qa import QuestionAnsweringTrainer
from extractor_data_loader import DataAssembler
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
)
from utils_qa import check_no_error, postprocess_qa_predictions, set_seed, MRCPreprocessor

"""
[코드 구조 및 설정 설명]

1. Arguments 설정:
   - 실행 시 필요한 설정값들은 ./arguments.py 또는 transformers/training_args.py에서 확인할 수 있습니다.
   - --help 플래그를 통해 커맨드라인에서 확인 가능합니다.
   - HfArgumentParser를 통해 ModelArguments, DataTrainingArguments, TrainingArguments를 파싱합니다.

2. Logging 및 WandB:
   - WandB 프로젝트는 'QDQA'로 설정되며, run_name이 없으면 output_dir의 마지막 경로명을 사용합니다.
   - Transformers logger의 verbosity를 설정하여 학습 진행 상황을 모니터링합니다.

3. Tokenizer:
   - 'use_fast=True' 설정을 통해 Rust로 구현된 빠른 속도의 tokenizer를 사용합니다.
   - 모델에 따라 fast tokenizer를 지원하지 않는 경우도 있으니 확인이 필요합니다.

4. 데이터 전처리 (Preprocessing):
   - 질문(Question)과 지문(Context)을 합쳐서 모델 입력으로 사용합니다.
   - Padding 옵션은 tokenizer 설정(padding_side)에 따라 (question|context) 또는 (context|question) 순서로 결정됩니다.
   - 긴 문서(Context)의 경우 stride를 사용하여 여러 개의 feature로 나누어 처리합니다 (doc_stride).
   - 정답(Answer) 위치를 찾기 위해 offset_mapping을 사용합니다.

5. 평가 (Evaluation):
   - 예측 결과(Start/End logits)를 후처리(post_processing_function)하여 텍스트 형태의 정답으로 변환합니다.
   - SQuAD metric을 사용하여 성능을 평가합니다.
"""

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)


    
    if any(arg.startswith("--seed") for arg in sys.argv):
        logger.info(f"시드 설정 {training_args.seed}")
        set_seed(training_args.seed)

    
    logger.info(f"데이터셋 로드 및 병합 (Source: {data_args.train_datasets})...")
    data_assembler = DataAssembler(data_args)
    datasets = data_assembler.get_datasets()
    
    
    logger.info("Config, Tokenizer, Model 로드...")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # WandB 설정
    training_args.report_to = ["wandb"]
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "QDQA"
    if training_args.run_name is None:
        training_args.run_name = training_args.output_dir.split("/")[-1]

    # 최종 설정 확인용 요약 출력 (Training 전 사용자 확인용)
    print("\n" + "=" * 50)
    print(" [ RUN CONFIGURATION SUMMARY ]")
    print(f" | Model          : {model_args.model_name_or_path}")
    print(f" | Dataset        : {data_args.dataset_name}")
    print(f" | Train Samples  : {len(datasets['train']) if 'train' in datasets else 0}")
    print(f" | Output Dir     : {training_args.output_dir}")
    print(f" | Run Name       : {training_args.run_name}")
    print(f" | Batch Size     : {training_args.per_device_train_batch_size}")
    print(f" | Learning Rate  : {training_args.learning_rate}")
    print(f" | Epochs         : {training_args.num_train_epochs}")
    print(f" | Seed           : {training_args.seed}")
    print(f" | Train Sources  : {data_args.train_datasets}")
    print("=" * 50 + "\n")

    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # 데이터셋 전처리 설정
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Preprocessor 인스턴스 생성
    processor = MRCPreprocessor(
        tokenizer=tokenizer,
        data_args=data_args,
        column_names=column_names,
        max_seq_length=max_seq_length,
    )

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        
        logger.info("학습 데이터 전처리...")
        train_dataset = train_dataset.map(
            processor.prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        logger.info("검증 데이터 전처리...")
        eval_dataset = eval_dataset.map(
            processor.prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

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
                {"id": ex["id"], "answers": ex[processor.answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        metrics = metric.compute(predictions=p.predictions, references=p.label_ids)
        return {f"eval_{k}": v for k, v in metrics.items()}

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path) and not training_args.overwrite_output_dir:
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        
        logger.info("학습 시작...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** 학습 결과 *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    if training_args.do_eval:
        logger.info("평가 시작...")
        logger.info("*** 평가 수행 ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

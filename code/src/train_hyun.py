import logging
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_dataset, concatenate_datasets
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions

# [수정 1] RetroReader 모델 Import (QA 모델에 검증기(Verifier) 기능을 추가하기 위해 필요)
from joint_model_with_retro import RetroReaderJointModel

logger = logging.getLogger(__name__)


def main():
    # 학습 및 데이터 관련 Argument (인자)를 파싱합니다.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- WANDB 설정 시작 ---
    # 학습 과정 로깅을 위해 Weights & Biases (WandB) 사용을 설정합니다.
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "QDQA"      # WandB 프로젝트 이름
    os.environ["WANDB_ENTITY"] = "hakiful-ai"  # 팀(Organization) 이름

    # run name이 지정되지 않은 경우, output_dir 이름을 사용합니다.
    if training_args.run_name is None:
        training_args.run_name = training_args.output_dir.split("/")[-1]
    # --- WANDB 설정 종료 ---

    # 모델 및 데이터 경로 정보 출력
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # 기본 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Transformers logger의 verbosity (로그 상세도) 설정
    logger.info("Training/evaluation parameters %s", training_args)

    # 재현성 확보를 위해 난수 시드(Seed)를 고정합니다.
    set_seed(training_args.seed)

    # 데이터셋을 로드합니다.
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # --- KorQuad 데이터셋 병합 로직 ---
    if data_args.add_korquad:
        print("Adding KorQuad v1 dataset...")
        korquad = load_dataset("squad_kor_v1")
        kq_train = korquad["train"]

        if "train" in datasets:
            org_train = datasets["train"]

            # 기존 데이터셋과 KorQuad 데이터셋 간의 공통 칼럼을 찾습니다.
            common_columns = [
                col for col in org_train.column_names if col in kq_train.column_names
            ]

            # 두 데이터셋 모두 공통 칼럼만 선택하여 피처를 일치시킵니다.
            kq_train = kq_train.select_columns(common_columns)
            org_train_filtered = org_train.select_columns(common_columns)
            
            # 피처 타입을 일치시킨 후, 두 데이터셋을 병합(Concatenate)합니다.
            kq_train = kq_train.cast(org_train_filtered.features)
            combined_train = concatenate_datasets([org_train_filtered, kq_train])
            datasets["train"] = combined_train
            print(f"Added KorQuad. New train size: {len(datasets['train'])}")
    # --- KorQuad 데이터셋 병합 로직 종료 ---

    # Pretrained 모델, 설정(Config), 토크나이저를 로드합니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True, # Rust 기반의 빠른 토크나이저 사용
    )
    
    # [수정 2] 모델 로드 및 RetroReader 래핑
    # 1. 일반 QA 모델(Reader)을 로드합니다.
    base_model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    
    # 2. 로드된 Reader를 RetroReaderJointModel로 감싸 Verifier 기능을 추가합니다.
    print("Initialize Retro-Reader Joint Model (Fast Verification)...")
    model = RetroReaderJointModel(
        reader=base_model,
        config=config
    )

    # 객체 타입 출력
    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # 학습 또는 평가를 실행합니다.
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

    # 데이터셋의 칼럼 이름을 설정합니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # 토크나이저의 Padding 방향을 설정합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류를 체크하고 최대 시퀀스 길이를 가져옵니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # --- 학습 데이터 전처리 함수 ---
    def prepare_train_features(examples):
        # 질문과 문맥(Context)을 토큰화합니다 (truncation, padding, stride 적용).
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,  # RoBERTa 모델 사용 시 False
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 오버플로우 토큰을 원본 샘플에 매핑하기 위한 정보를 가져옵니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        # RoBERTa 사용 시 token_type_ids가 생성되었다면 제거합니다.
        if "token_type_ids" in tokenized_examples:
            del tokenized_examples["token_type_ids"]

        # 정답 위치(Start/End) 레이블을 저장할 리스트를 초기화합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        # 각 토큰화된 인스턴스에 대해 정답 레이블을 계산합니다.
        for i, offsets in enumerate(offset_mapping):
            # ... (정답 위치 계산 로직 생략 - 원본 코드 유지) ...
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            if len(answers["answer_start"]) == 0:
                # 정답이 없는 경우, CLS 토큰 인덱스를 레이블로 설정합니다.
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 현재 span을 벗어났는지 확인
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    # 벗어났다면 CLS 토큰을 레이블로 설정합니다.
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # 정답이 span 내에 있다면, 정답 위치의 토큰 인덱스를 찾습니다.
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    # --- 학습 데이터 전처리 함수 종료 ---

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # --- 평가 데이터 전처리 함수 ---
    def prepare_validation_features(examples):
        # 질문과 문맥(Context)을 토큰화합니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,  # RoBERTa 모델 사용 시 False
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # RoBERTa 사용 시 token_type_ids 제거
        if "token_type_ids" in tokenized_examples:
            del tokenized_examples["token_type_ids"]

        # 평가를 위해 example_id와 offset_mapping을 저장합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Context 부분이 아닌 토큰의 offset_mapping을 None으로 설정합니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples
    # --- 평가 데이터 전처리 함수 종료 ---

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator (데이터 패딩을 담당)
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # --- Post-processing 함수 ---
    def post_processing_function(examples, features, predictions, training_args):
        # [수정 3] Retro-Reader의 3개 Tuple 출력 처리 (Start, End, Verify)
        # 텍스트 추출 함수(postprocess_qa_predictions)에는 Start/End Logit 2개만 넘깁니다.
        span_predictions = predictions
        if isinstance(predictions, tuple) and len(predictions) >= 2:
            span_predictions = (predictions[0], predictions[1])

        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions_text = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=span_predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        
        # Metric 계산을 위해 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions_text.items()
        ]
        
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            # 최종 평가 결과 (predictions에는 Verifier Logit까지 포함됨)
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    # 평가 메트릭 (SQuAD) 로드
    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        metrics = metric.compute(predictions=p.predictions, references=p.label_ids)
        # Trainer가 요구하는 "eval_" 접두사 추가
        return {f"eval_{k}": v for k, v in metrics.items()}

    # Trainer 초기화
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

    # --- Training 시작 ---
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path) and not training_args.overwrite_output_dir:
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # [수정 4] 모델 분리 저장 (Verifier를 분리하여 저장해야 추론 시 활용 가능)
        print("Saving Retro-Reader models separately...")
        output_dir = training_args.output_dir
        
        # 1. Reader 저장 (Hugging Face 포맷 - 모델의 질문/답변 추출 부분)
        reader_save_path = os.path.join(output_dir, "reader")
        if hasattr(model, "module"): # 분산 학습(DDP) 환경 고려
            model.module.reader.save_pretrained(reader_save_path)
        else:
            model.reader.save_pretrained(reader_save_path)
            
        # 2. Verifier 저장 (PyTorch State Dict 포맷 - 모델의 검증 부분)
        verifier_save_path = os.path.join(output_dir, "retro_verifier.pth")
        if hasattr(model, "module"): # 분산 학습(DDP) 환경 고려
            torch.save(model.module.verifier.state_dict(), verifier_save_path)
        else:
            torch.save(model.verifier.state_dict(), verifier_save_path)
            
        tokenizer.save_pretrained(reader_save_path)
        print(f"✅ Saved Reader to: {reader_save_path}")
        print(f"✅ Saved Verifier to: {verifier_save_path}")

        # 학습 결과 로깅 및 저장
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Trainer 상태 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # --- Evaluation 시작 ---
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
"""
Open-Domain Question Answering을 수행하는 Inference 코드입니다.

[코드 구조 및 설정 설명]

1. Arguments 및 설정:
   - 대부분의 설정은 train.py와 유사하며, retrieval 및 predict 관련 로직이 추가되었습니다.
   - ./arguments.py 또는 transformers/training_args.py에서 가능한 인자를 확인할 수 있습니다.

2. Retrieval (검색):
   - QdrantHybridRetrieval을 사용하여 쿼리에 맞는 문서를 검색합니다.
   - Dense(Embedding) 검색과 Sparse(BM25) 검색을 결합하여 사용합니다 (alpha 값으로 가중치 조절).

3. Inference (추론):
   - 검색된 문서를 Context로 하여 MRC 모델이 정답을 찾습니다.
   - do_predict: 정답이 없는 Test set에 대한 추론.
   - do_eval: 정답이 있는 Validation set에 대한 평가.
"""

import logging
import sys
import os
from typing import Callable, Dict, List, NoReturn, Tuple

import evaluate
import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
from retrieval_qdrant_final import QdrantHybridRetrieval
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

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # 요약 정보 수집
    summary = []
    summary.append("=" * 30)
    summary.append(f"Model: {model_args.model_name_or_path}")
    summary.append(f"Dataset: {data_args.dataset_name}")
    summary.append(f"Output Dir: {training_args.output_dir}")

    # WandB 설정
    training_args.report_to = ["wandb"]
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "QDQA"
    if training_args.run_name is None:
        training_args.run_name = training_args.output_dir.split("/")[-1]
    summary.append(f"Run Name: {training_args.run_name}")
    summary.append("=" * 30)

    # 요약 정보 출력
    print("\n".join(summary))

    logger.info("Training/evaluation parameters %s", training_args)

    # Seed 고정
    set_seed(training_args.seed)

    # 데이터셋 로드
    datasets = load_from_disk(data_args.dataset_name)
    logger.info(f"Loaded datasets: {datasets}")

    # Config, Tokenizer, Model 로드
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

    # Retrieval 실행
    if data_args.eval_retrieval:
        logger.info("Running Retrieval...")
        datasets = run_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )

    # MRC 실행 (Eval or Predict)
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "raw/data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    retriever = QdrantHybridRetrieval(
        data_path=data_path,
        context_path=context_path,
        use_fp16=training_args.fp16,
    )

    alpha = getattr(data_args, "alpha", getattr(data_args, "dense_weight", 0.5))

    df = retriever.retrieve(
        datasets["validation"],
        topk=data_args.top_k_retrieval,
        alpha=alpha,
    )

    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    else:
        if "answers" in df.columns:
            f = Features(
                {
                    "answers": Sequence(
                        feature={
                            "text": Value(dtype="string", id=None),
                            "answer_start": Value(dtype="int32", id=None),
                        },
                        length=-1,
                        id=None,
                    ),
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )
        else:
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )

    df = df[list(f.keys())]
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"

    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Sub-directory creation based on mode
    if training_args.do_eval:
        training_args.output_dir = os.path.join(training_args.output_dir, "eval_pred")
    elif training_args.do_predict:
        training_args.output_dir = os.path.join(training_args.output_dir, "test")

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
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

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    logger.info("Initializing Trainer...")
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

    logger.info("*** Evaluate ***")

    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        logger.info("Prediction completed. Results saved via post-processing.")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()

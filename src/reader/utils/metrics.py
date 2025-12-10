"""
평가 메트릭 모듈
"""
import os
from typing import Any, Dict, Tuple

import evaluate
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast, TrainingArguments, EvalPrediction
from transformers.trainer_utils import get_last_checkpoint

from src.utils.arguments import DataTrainingArguments


def compute_squad_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    SQuAD 메트릭 (EM, F1) 계산
    
    Args:
        p: EvalPrediction (predictions, label_ids)
    
    Returns:
        메트릭 딕셔너리
    """
    metric = evaluate.load("squad")
    return metric.compute(predictions=p.predictions, references=p.label_ids)


def compute_metrics_with_prefix(p: EvalPrediction) -> Dict[str, float]:
    """eval_ prefix가 붙은 메트릭 반환"""
    metrics = compute_squad_metrics(p)
    return {f"eval_{k}": v for k, v in metrics.items()}


def check_no_error(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> Tuple[Any, int]:
    """
    학습 전 에러 체크
    
    Returns:
        (last_checkpoint, max_seq_length)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # last checkpoint 찾기
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Tokenizer check
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer."
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    
    return last_checkpoint, max_seq_length

"""
데이터 전처리 모듈 (Training/Validation Features 생성)
"""
from typing import Dict, List, Any

from transformers import PreTrainedTokenizer


class DataProcessor:
    """QA 데이터 전처리기"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        pad_to_max_length: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.pad_to_max_length = pad_to_max_length
        self.pad_on_right = tokenizer.padding_side == "right"
    
    def prepare_train_features(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        학습용 features를 생성합니다.
        
        Args:
            examples: 원본 데이터 배치
        
        Returns:
            토큰화된 features
        """
        question_column = "question"
        context_column = "context"
        answer_column = "answers"
        
        tokenized_examples = self.tokenizer(
            examples[question_column if self.pad_on_right else context_column],
            examples[context_column if self.pad_on_right else question_column],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            padding="max_length" if self.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
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

    def prepare_validation_features(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        평가용 features를 생성합니다.
        
        Args:
            examples: 원본 데이터 배치
        
        Returns:
            토큰화된 features
        """
        question_column = "question"
        context_column = "context"
        
        tokenized_examples = self.tokenizer(
            examples[question_column if self.pad_on_right else context_column],
            examples[context_column if self.pad_on_right else question_column],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            padding="max_length" if self.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples

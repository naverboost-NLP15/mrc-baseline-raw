"""
MRC Reader 모델 래퍼
"""
from typing import Optional

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class MRCReader:
    """MRC Reader 모델 래퍼"""
    
    def __init__(
        self,
        model_name_or_path: str = "klue/bert-base",
        config_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        use_fast_tokenizer: bool = True,
    ):
        """
        Args:
            model_name_or_path: 모델 이름 또는 경로
            config_name: Config 이름 (None이면 model_name_or_path 사용)
            tokenizer_name: Tokenizer 이름 (None이면 model_name_or_path 사용)
            use_fast_tokenizer: Fast tokenizer 사용 여부
        """
        self.model_name_or_path = model_name_or_path
        
        # Config 로드
        self.config = AutoConfig.from_pretrained(
            config_name if config_name else model_name_or_path
        )
        
        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            use_fast=use_fast_tokenizer,
        )
        
        # Model 로드
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=self.config,
        )
    
    def get_model(self) -> PreTrainedModel:
        """모델 반환"""
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """토크나이저 반환"""
        return self.tokenizer
    
    def get_config(self) -> AutoConfig:
        """Config 반환"""
        return self.config
    
    @property
    def pad_on_right(self) -> bool:
        """Padding이 오른쪽인지 확인"""
        return self.tokenizer.padding_side == "right"
    
    @property
    def model_max_length(self) -> int:
        """모델 최대 길이"""
        return self.tokenizer.model_max_length

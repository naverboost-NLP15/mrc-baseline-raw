"""
Random Seed 고정 유틸리티
"""
import random

import numpy as np
import torch
from transformers import is_torch_available


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    전체 random seed를 고정합니다.
    
    Args:
        seed: 고정할 seed 값
        deterministic: cudnn deterministic 모드 (학습 속도가 느려질 수 있음)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

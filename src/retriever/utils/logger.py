"""
로깅 유틸리티
"""
import logging
import sys
from typing import Optional


def get_logger(
    name: str = __name__,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt: str = "%m/%d/%Y %H:%M:%S",
) -> logging.Logger:
    """
    로거를 생성하고 반환합니다.
    
    Args:
        name: 로거 이름
        level: 로깅 레벨
        format_str: 로그 포맷
        datefmt: 날짜 포맷
    
    Returns:
        설정된 Logger 인스턴스
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_str, datefmt=datefmt))
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """전역 로깅 설정"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=level,
    )

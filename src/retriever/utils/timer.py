"""
타이머 유틸리티
"""
import time
from contextlib import contextmanager
from typing import Optional


@contextmanager
def timer(name: str = "Task"):
    """
    작업 시간을 측정하는 context manager
    
    Usage:
        with timer("Encoding"):
            # do something
    """
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    print(f"[{name}] done in {elapsed:.3f} s")


class Timer:
    """재사용 가능한 타이머 클래스"""
    
    def __init__(self, name: str = "Task"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self) -> "Timer":
        """타이머 시작"""
        self.start_time = time.time()
        return self
    
    def stop(self) -> float:
        """타이머 종료 및 경과 시간 반환"""
        self.end_time = time.time()
        return self.elapsed
    
    @property
    def elapsed(self) -> float:
        """경과 시간 (초)"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def __enter__(self) -> "Timer":
        return self.start()
    
    def __exit__(self, *args) -> None:
        elapsed = self.stop()
        print(f"[{self.name}] done in {elapsed:.3f} s")

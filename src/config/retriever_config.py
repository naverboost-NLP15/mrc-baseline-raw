"""
Retriever 관련 설정
"""

# Retrieval Settings
DEFAULT_TOP_K = 10
DEFAULT_TOP_N = 50  # Reranking 전 후보 수
DEFAULT_ALPHA = 0.5  # Dense weight (0.0 = Sparse Only, 1.0 = Dense Only)

# Sparse Type Options: "splade", "bm25", "custom_sparse"
DEFAULT_SPARSE_TYPE = "splade"

# BM25 Parameters
BM25_K1 = 1.2
BM25_B = 0.75

# RRF Fusion Parameter
RRF_K = 60

# Reranker Settings
USE_RERANKER = True
USE_FP16 = False

# Batch Settings
SEARCH_BATCH_SIZE = 64
ENCODE_BATCH_SIZE = 16

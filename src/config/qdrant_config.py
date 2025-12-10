"""
Qdrant 연결 설정
"""

# Qdrant Server Configuration
QDRANT_HOST = "lori2mai11ya.asuscomm.com"
QDRANT_PORT = 6333
QDRANT_API_KEY = "boostcamp"
QDRANT_HTTPS = False
QDRANT_TIMEOUT = 60

# Collection Settings
DEFAULT_COLLECTION_NAME = "hybird_collection_v1"

# Model Names
DEFAULT_DENSE_MODEL = "telepix/PIXIE-Spell-Preview-1.7B"
DEFAULT_SPARSE_MODEL = "telepix/PIXIE-Splade-Preview"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Chunking Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

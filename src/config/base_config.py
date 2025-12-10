"""
공통 설정 (seed, wandb 등)
"""
import os

# Random Seed
SEED = 2024
DETERMINISTIC = False

# WandB 설정
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "QDQA")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# 데이터 경로
DATA_PATH = "raw/data"
CONTEXT_PATH = "wikipedia_documents.json"
TRAIN_DATASET_PATH = "raw/data/train_dataset"
TEST_DATASET_PATH = "raw/data/test_dataset"

# 출력 경로
OUTPUT_DIR = "output"
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

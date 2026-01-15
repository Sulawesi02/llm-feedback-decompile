import os
from pathlib import Path

# 目录路径
PROJECT_ROOT = Path("/app")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "exebench"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
EVAL_DIR = PROJECT_ROOT / "eval"

TEMP_DIR = Path("/tmp/workdir")

# 数据集文件路径
TEST_DATA_FILE_PATH = PROCESSED_DATA_DIR / "test_asm_to_c_dedup.jsonl"
TRAIN_DATA_FILE_PATH = PROCESSED_DATA_DIR / "train_asm_to_c_dedup.jsonl"
VALID_DATA_FILE_PATH = PROCESSED_DATA_DIR / "valid_asm_to_c_dedup.jsonl"

# 模型路径
BASE_MODEL_NAME = "Qwen2.5-Coder-7B-Instruct"
BASE_MODEL_DIR_PATH = MODEL_DIR / "base_model" / BASE_MODEL_NAME
LORA_CHECKPOINTS_DIR_PATH = MODEL_DIR / "lora_checkpoints"
MERGED_MODEL_DIR_PATH = MODEL_DIR / "merged_model"

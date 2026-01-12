import os
from pathlib import Path

# 目录路径
PROJECT_ROOT = Path("/app")
DATA_DIR = PROJECT_ROOT / "data" / "exebench"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
EVAL_DIR = PROJECT_ROOT / "eval"
TEMP_DIR = Path("/tmp/workdir")

# 确保关键目录存在
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 数据集文件路径
TEST_DATA_PATH = DATA_DIR / "test" / "test_machine_to_c.jsonl"
TRAIN_DATA_PATH = DATA_DIR / "train" / "train_machine_to_c.jsonl"
VALID_DATA_PATH = DATA_DIR / "valid" / "valid_machine_to_c.jsonl"

# 模型路径
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_MODEL_PATH = MODEL_DIR / "base_model" / BASE_MODEL_NAME
LORA_CHECKPOINTS_DIR = MODEL_DIR / "lora_checkpoints"
MERGED_MODEL_DIR = MODEL_DIR / "merged_model"
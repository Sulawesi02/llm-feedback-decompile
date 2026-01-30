import os
import platform
import tempfile
from pathlib import Path

# 自动获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据集目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
DEDUPLICATED_DATA_DIR = DATA_DIR / "deduplicated_data"
TRAIN_DATA = DEDUPLICATED_DATA_DIR / "train_data.jsonl"
VALID_DATA = DEDUPLICATED_DATA_DIR / "valid_data.jsonl"
TEST_DATA = DEDUPLICATED_DATA_DIR / "test_data.jsonl"
DPO_DATA_DIR = DATA_DIR / "dpo_data"

# 模型目录
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_NAME = "Qwen2.5-Coder-7B-Instruct"
BASE_MODEL_DIR = MODEL_DIR / "base_models"
SFT_ADAPTER_DIR = MODEL_DIR / "sft_adapter"
DPO_ADAPTER_DIR = MODEL_DIR / "dpo_adapter"
VERSIONS = [ # (版本号, 数据比例)
    ("v1", 0.01), 
    ("v2", 0.02),
    ("v3", 0.05),
]

# 工作目录
TEMP_DIR = Path(tempfile.gettempdir()) / "workdir"

# Offload 目录
OFFLOAD_ROOT = Path(tempfile.gettempdir()) / "offload"

# 评估目录
EVAL_DIR = PROJECT_ROOT / "eval"



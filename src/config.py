import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据集目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
DPO_DATA_DIR = DATA_DIR / "dpo_data"

# 配置目录
CONFIG_DIR = PROJECT_ROOT / "configs"

# 模型目录
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 适配器目录
ADAPTER_DIR = PROJECT_ROOT / "adapter"
SFT_DIR = ADAPTER_DIR / "sft"
DPO_DIR = ADAPTER_DIR / "dpo"

# 评估目录
EVAL_DIR = PROJECT_ROOT / "eval"

MAX_PROMPT_TOKENS = 4096 # 最大提示 tokens 数
MAX_GEN_TOKENS = 512 # 最大生成 tokens 数

# HF 镜像源
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

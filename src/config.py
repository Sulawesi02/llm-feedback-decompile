import tempfile
import torch
from pathlib import Path
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# 自动获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据集目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
DEDUP_DATA_DIR = DATA_DIR / "dedup_data"
TRAIN_DATA = DEDUP_DATA_DIR / "train_data.jsonl"
VALID_DATA = DEDUP_DATA_DIR / "valid_data.jsonl"
TEST_DATA = DEDUP_DATA_DIR / "test_data.jsonl"
SFT_DATA_DIR = DATA_DIR / "sft_data"
DPO_DATA_DIR = DATA_DIR / "dpo_data"

# 模型目录
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_NAME = "Qwen2.5-Coder-7B-Instruct"
BASE_MODEL_DIR = MODEL_DIR / "base_models"
SFT_ADAPTER_DIR = MODEL_DIR / "sft_adapter"
DPO_ADAPTER_DIR = MODEL_DIR / "dpo_adapter"
VERSIONS = [ # (版本号, 数据比例)
    ("v1", 0.10), 
    ("v2", 0.50),
    ("v3", 1.00),
]

# 工作目录
TEMP_DIR = Path(tempfile.gettempdir()) / "workdir"

# 评估目录
EVAL_DIR = PROJECT_ROOT / "eval"

# 量化配置
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA 配置
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    use_rslora=True,
)
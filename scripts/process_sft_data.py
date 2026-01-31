import sys
import json
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_DATA, 
    VALID_DATA, 
    SFT_DATA_DIR,
    MODEL_NAME,
    BASE_MODEL_DIR,
    VERSIONS,
)
from src.utils import extract_compilation_data

SFT_DATA_RATIO = 0.05 # SFT 数据生成采样比例

def main():
    if not TRAIN_DATA.exists():
        print(f"错误: 训练集不存在: {TRAIN_DATA}")
        return
    if not VALID_DATA.exists():
        print(f"错误: 验证集不存在: {VALID_DATA}")
        return
    SFT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_NAME:
        print("错误: 模型名称未配置")
        return
    if not BASE_MODEL_DIR.exists():
        print(f"错误: 基座模型目录不存在: {BASE_MODEL_DIR}")
        return
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    if not base_model_path.exists():
        print(f"错误: 基座模型不存在: {base_model_path}")
        return
    if not VERSIONS:
        print("错误: 版本未配置")
        return

    train_output_path = SFT_DATA_DIR / "train_data.jsonl"
    valid_output_path = SFT_DATA_DIR / "valid_data.jsonl"
    
    train_exists = train_output_path.exists()
    valid_exists = valid_output_path.exists()

    if train_exists and valid_exists:
        print("SFT 训练数据和验证数据均已存在，跳过处理。")
        return

    if not train_exists:
        print("生成训练数据...")
        raw_train = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
        print(f"原始训练数据: {len(raw_train)} 条")
        sampled_train = raw_train.shuffle(seed=42).select(range(int(len(raw_train) * SFT_DATA_RATIO)))
        print(f"采样训练数据: {len(sampled_train)} 条")
        
        print("格式化训练数据...")
        formatted_train = []
        for item in tqdm(sampled_train, desc="生成 SFT 训练数据进度"):
            for arch, asm, c_code, _ in extract_compilation_data(item):
                entry = {
                    "instruction": f"根据给定的目标架构({arch})和汇编代码({asm})，输出一个在语义上完全等价的 C 函数实现",
                    "outputs": c_code,
                }
                formatted_train.append(entry)

        print("保存训练数据...")
        with open(train_output_path, "w", encoding="utf-8") as f:
            for entry in formatted_train:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"保存 {len(formatted_train)} 条训练数据到 {train_output_path}")
    else:
        print(f"训练数据已存在，跳过: {train_output_path}")

    if not valid_exists:
        raw_valid = load_dataset("json", data_files=str(VALID_DATA), split="train")
        print(f"原始验证数据: {len(raw_valid)} 条")
        sampled_valid = raw_valid.shuffle(seed=42).select(range(int(len(raw_valid) * SFT_DATA_RATIO)))
        print(f"采样验证数据: {len(sampled_valid)} 条")

        print("格式化验证数据...")
        formatted_valid = []
        for item in tqdm(sampled_valid, desc="生成验证数据进度"):
            for arch, asm, c_code, _ in extract_compilation_data(item):
                entry = {
                    "instruction": f"根据给定的目标架构({arch})和汇编代码({asm})，输出一个在语义上完全等价的 C 函数实现",
                    "outputs": c_code,
                }
                formatted_valid.append(entry)

        print("保存验证数据...")
        with open(valid_output_path, "w", encoding="utf-8") as f:
            for entry in formatted_valid:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"保存 {len(formatted_valid)} 条验证数据到 {valid_output_path}")
    else:
        print(f"验证数据已存在，跳过: {valid_output_path}")

if __name__ == "__main__":
    main()

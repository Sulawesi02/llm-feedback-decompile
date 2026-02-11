import sys
import json
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    PROCESSED_DATA_DIR,
    SFT_DATA_DIR,
)

def main():
    train_input = PROCESSED_DATA_DIR / "train_data.jsonl"
    valid_input = PROCESSED_DATA_DIR / "valid_data.jsonl"
    SFT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_output = SFT_DATA_DIR / "train_data.jsonl"
    valid_output = SFT_DATA_DIR / "valid_data.jsonl"
    
    if train_output.exists() and valid_output.exists():
        print("SFT 训练数据和验证数据均已存在，跳过处理。")
        return

    if train_output.exists():
        print(f"训练数据已存在，跳过: {train_output}")
    else:
        print(f"加载训练数据集: {train_input}")
        train_data = load_dataset("json", data_files=str(train_input), split="train")
        print("格式化训练数据...")
        formatted_train = []
        for item in tqdm(train_data, desc="生成 SFT 训练数据进度"):
            entry = {
                "instruction": f"根据目标架构x86和给定的汇编代码({item['asm']})，输出一个在语义上完全等价的 C 函数实现",
                "response": item["func"],
            }
            formatted_train.append(entry)

        print("保存训练数据...")
        with open(train_output, "w", encoding="utf-8") as f:
            for entry in formatted_train:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"保存 {len(formatted_train)} 条训练数据到 {train_output}")

    if valid_output.exists():
        print(f"验证数据已存在，跳过: {valid_output}")
    else:
        print(f"加载验证数据集: {valid_input}")
        valid_data = load_dataset("json", data_files=str(valid_input), split="train")
        print(f"原始验证数据: {len(valid_data)} 条")
        
        print("格式化验证数据...")
        formatted_valid = []
        for item in tqdm(valid_data, desc="生成验证数据进度"):
            entry = {
                "instruction": f"根据目标架构x86和给定的汇编代码({item['asm']})，输出一个在语义上完全等价的 C 函数实现",
                "response": item["func"],
            }
            formatted_valid.append(entry)

        print("保存验证数据...")
        with open(valid_output, "w", encoding="utf-8") as f:
            for entry in formatted_valid:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"保存 {len(formatted_valid)} 条验证数据到 {valid_output}")

if __name__ == "__main__":
    main()

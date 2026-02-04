import sys
import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_DATA, 
    VALID_DATA, 
    DPO_DATA_DIR, 
)
from src.utils import extract_data

DPO_DATA_RATIO = 0.05 # DPO 数据生成采样比例

def generate_bad_code(c_code: str) -> str:
    """
    通过简单的规则扰动生成错误代码
    """
    lines = c_code.splitlines()
    if not lines:
        return c_code
        
    mutation_type = random.choice(["delete", "swap", "modify_op", "modify_num"])
    
    try:
        if mutation_type == "delete" and len(lines) > 2:
            # 随机删除一行非空行（避开开头结尾）
            idx = random.randint(1, len(lines) - 2)
            lines.pop(idx)
            
        elif mutation_type == "swap" and len(lines) > 3:
            # 随机交换两行
            idx1 = random.randint(1, len(lines) - 2)
            idx2 = random.randint(1, len(lines) - 2)
            lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
            
        elif mutation_type == "modify_op":
            # 替换运算符
            code_str = "\n".join(lines)
            ops = [("+", "-"), ("*", "/"), ("==", "!="), (">", "<"), ("&", "|")]
            op = random.choice(ops)
            if op[0] in code_str:
                code_str = code_str.replace(op[0], op[1], 1)
                return code_str
                
        elif mutation_type == "modify_num":
            # 修改数字
            code_str = "\n".join(lines)
            import re
            nums = re.findall(r'\b\d+\b', code_str)
            if nums:
                target = random.choice(nums)
                new_num = str(int(target) + random.randint(1, 10))
                code_str = code_str.replace(target, new_num, 1)
                return code_str
                
    except Exception:
        pass
        
    return "\n".join(lines)

def main():
    if not TRAIN_DATA.exists():
        print(f"错误: 训练集不存在: {TRAIN_DATA}")
        return
    if not VALID_DATA.exists():
        print(f"错误: 验证集不存在: {VALID_DATA}")
        return
    DPO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_output_path = DPO_DATA_DIR / "train_data.jsonl"
    valid_output_path = DPO_DATA_DIR / "valid_data.jsonl"

    train_exists = train_output_path.exists()
    valid_exists = valid_output_path.exists()

    if train_exists and valid_exists:
        print("DPO 训练数据和验证数据均已存在，跳过处理。")
        return

    if not train_exists:
        print("生成训练数据...")
        raw_train = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
        print(f"原始训练数据: {len(raw_train)} 条")
        sampled_train = raw_train.shuffle(seed=42).select(range(int(len(raw_train) * DPO_DATA_RATIO)))
        print(f"采样训练数据: {len(sampled_train)} 条")

        print("格式化训练数据...")
        formatted_train = []
        for item in tqdm(sampled_train, desc="生成 DPO 训练数据进度"):
            for c_code, arch, asm, _ in extract_data(item):
                rejected_code = generate_bad_code(c_code)
                
                entry = {
                    "prompt": f"根据给定的目标架构({arch})和汇编代码({asm})，输出一个在语义上完全等价的 C 函数实现",
                    "chosen": c_code,
                    "rejected": rejected_code
                }
                formatted_train.append(entry)

        print("保存训练数据...")
        with open(train_output_path, "w", encoding="utf-8") as f:
            for entry in formatted_train:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"保存 {len(formatted_train)} 条训练数据到 {train_output_path}")
    else:
        print("训练数据已存在，跳过生成。")
    
    if valid_exists:
        print(f"验证数据已存在，跳过: {valid_output_path}")
    else:
        print("生成验证数据 (Rejected Samples)...")
        print("加载验证数据集...")
        raw_valid = load_dataset("json", data_files=str(VALID_DATA), split="train")
        print(f"原始验证数据: {len(raw_valid)} 条")
        sampled_valid = raw_valid.shuffle(seed=42).select(range(int(len(raw_valid) * DPO_DATA_RATIO)))
        print(f"采样验证数据: {len(sampled_valid)} 条")

        formatted_valid = []
        for item in tqdm(sampled_valid, desc="生成 DPO 验证数据进度"):
            for c_code, arch, asm, _ in extract_data(item):
                rejected_code = generate_bad_code(c_code)
                
                entry = {
                    "prompt": f"根据给定的目标架构({arch})和汇编代码({asm})，输出一个在语义上完全等价的 C 函数实现",
                    "chosen": c_code,
                    "rejected": rejected_code
                }
                formatted_valid.append(entry)
    
        print("保存验证数据...")
        with open(valid_output_path, "w", encoding="utf-8") as f:
            for entry in formatted_valid:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"保存 {len(formatted_valid)} 条验证数据到 {valid_output_path}")

if __name__ == "__main__":
    main()

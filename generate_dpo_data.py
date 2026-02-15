import json
import random
from tqdm import tqdm
from datasets import load_dataset

from src.config import (
    PROCESSED_DATA_DIR,
    DPO_DATA_DIR,
)

def generate_bad_code(func: str) -> str:
    """
    通过简单的规则扰动生成错误代码
    """
    lines = func.splitlines()
    if not lines:
        return func
        
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
    train_input = PROCESSED_DATA_DIR / "train_data.jsonl"
    valid_input = PROCESSED_DATA_DIR / "valid_data.jsonl"

    if not train_input.exists():
        print(f"错误: 训练集不存在: {train_input}")
        return
    if not valid_input.exists():
        print(f"错误: 验证集不存在: {valid_input}")
        return
    
    train_output = DPO_DATA_DIR / "train_data.jsonl"
    valid_output = DPO_DATA_DIR / "valid_data.jsonl"

    if train_output.exists() and valid_output.exists():
        print("DPO 训练数据和验证数据均已存在，跳过处理。")
        return

    if train_output.exists():    
        print("训练数据已存在，跳过生成。")
    else:
        print(f"加载训练数据集: {train_input}")
        train_data = load_dataset("json", data_files=str(train_input), split="train")
        print("格式化训练数据...")
        formatted_train = []
        for item in tqdm(train_data, desc="生成 DPO 训练数据进度"):
            
            entry = {
                "prompt": f"根据给定的汇编代码({item['asm']})，输出一个在语义上完全等价的 C 函数实现",
                "chosen": item["func"],
                "rejected": generate_bad_code(item["func"])
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
        print("格式化验证数据...")
        formatted_valid = []
        for item in tqdm(valid_data, desc="生成 DPO 验证数据进度"):
            entry = {
                "prompt": f"根据给定的汇编代码({item['asm']})，输出一个在语义上完全等价的 C 函数实现",
                "chosen": item["func"],
                "rejected": generate_bad_code(item["func"])
            }
            formatted_valid.append(entry)
    
        print("保存验证数据...")
        with open(valid_output, "w", encoding="utf-8") as f:
            for entry in formatted_valid:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"保存 {len(formatted_valid)} 条验证数据到 {valid_output}")

if __name__ == "__main__":
    main()

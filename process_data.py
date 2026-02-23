import json
import hashlib
import random
import yaml
from tqdm import tqdm
from datasets import load_dataset

from src.config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    CONFIG_DIR,
)

def load_cfg():
    with open(CONFIG_DIR / "dataset.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

TRAIN_RATIO = float(load_cfg().get("process", {}).get("train_ratio"))

def normalize_text(text: str) -> list:
    """
    对文本进行归一化处理(替换换行符、制表符和多个空格为单个空格，最后按空格分割为 tokens 列表)
    """
    if not text:
        return []
    return text.replace("\r", "\n").replace("\t", " ").replace("\n", " ").split()

def make_shingles(tokens: list, k: int) -> list:
    """
    将 tokens 列表转换为长度为 k 的 shingles 序列
    """
    if not tokens:
        return []
    if len(tokens) <= k:
        return [" ".join(tokens)]
    shingles = []
    for i in range(len(tokens) - k + 1):
        shingles.append(" ".join(tokens[i : i + k]))
    return shingles

def compute_minhash(shingles: list, num_perm: int = 64) -> list:
    signature = [2**63 - 1] * num_perm
    for s in shingles:
        base = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        for i in range(num_perm):
            h = (base + i * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF
            if h < signature[i]:
                signature[i] = h
    return signature

def lsh_deduplicate(records: list, field: str = "func", num_perm: int = 64, bands: int = 8, shingle_size: int = 5) -> tuple:
    if not records:
        return [], []
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands")
    rows_per_band = num_perm // bands
    buckets = {}
    kept_indices = []
    kept_records = []
    
    print(f"开始去重处理 (字段: {field}, 总数: {len(records)})...")
    for idx, rec in enumerate(tqdm(records, desc="计算 MinHash")):
        text = str(rec.get(field, "") or "")
        tokens = normalize_text(text)
        shingles = make_shingles(tokens, shingle_size)
        if not shingles:
            continue
        sig = compute_minhash(shingles, num_perm=num_perm)
        is_duplicate = False
        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_key = (b, tuple(sig[start:end]))
            owner = buckets.get(band_key)
            if owner is not None:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_key = (b, tuple(sig[start:end]))
            buckets[band_key] = idx
        kept_indices.append(idx)
        kept_records.append(rec)
    return kept_records, kept_indices

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 处理训练集 (Decompile-Bench) 并划分验证集
    print("\n=== 处理训练数据 ===")
    train_input = RAW_DATA_DIR / "train_data.jsonl"
    train_output = PROCESSED_DATA_DIR / "train_data.jsonl"
    valid_output = PROCESSED_DATA_DIR / "valid_data.jsonl"
    
    if train_output.exists() and valid_output.exists():
        print(f"训练数据文件已存在: {train_output}")
        print(f"验证数据文件已存在: {valid_output}")
    else:
        train_data = load_dataset("json", data_files=str(train_input), split="train")
        print("格式化训练数据...")
        formatted_train = []
        for item in tqdm(train_data, desc="格式化训练数据"):
            data = {
                "func": item["code"],
                "asm": item["asm"]
            }
            formatted_train.append(data)
        
        print(f"  读取到 {len(formatted_train)} 条有效数据")
        
        # 去重
        if formatted_train:
            dedup_records, _ = lsh_deduplicate(formatted_train, field="func")
            print(f"  去重结果: {len(formatted_train)} -> {len(dedup_records)}")
            formatted_train = dedup_records
            
            # 划分训练集和验证集 (90% : 10%)
            random.seed(42)
            random.shuffle(formatted_train)
            split_idx = int(len(formatted_train) * TRAIN_RATIO)
            train_records = formatted_train[:split_idx]
            valid_records = formatted_train[split_idx:]
            
            print(f"  划分数据集: 训练集 {len(train_records)} 条, 验证集 {len(valid_records)} 条")
            
            with open(train_output, "w", encoding="utf-8") as f:
                for rec in train_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  已保存训练集: {train_output}")
            
            with open(valid_output, "w", encoding="utf-8") as f:
                for rec in valid_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  已保存验证集: {valid_output}")

    # 2. 处理测试集 (Decompile-Eval)
    print("\n=== 处理测试数据 ===")
    test_input = RAW_DATA_DIR / "test_data.jsonl"
    test_output = PROCESSED_DATA_DIR / "test_data.jsonl"
    
    if test_output.exists():
        print(f"测试数据文件已存在: {test_output}")
    else:
        valid_data = load_dataset("json", data_files=str(test_input), split="train")
        print("格式化测试数据...")
        formatted_train = []
        for item in tqdm(valid_data, desc="格式化测试数据"):
            data = {
                "func_dep": item["func_dep"],
                "func": item["func"],
                "test": item["test"],
                "asm": item["asm"],
            }
            formatted_train.append(data)
        
        print(f"  读取到 {len(formatted_train)} 条有效测试数据")
        
        # 去重
        if formatted_train:
            dedup_records, _ = lsh_deduplicate(formatted_train, field="func")
            print(f"  去重结果: {len(formatted_train)} -> {len(dedup_records)}")
            formatted_train = dedup_records
        
        with open(test_output, "w", encoding="utf-8") as f:
            for new_idx, rec in enumerate(formatted_train):
                out = {
                    "index": new_idx,
                    "func_dep": rec["func_dep"],
                    "func": rec["func"],
                    "test": rec["test"],
                    "asm": rec["asm"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  已保存测试集: {test_output}")
    
    print("\n数据集处理完成！")

if __name__ == "__main__":
    main()

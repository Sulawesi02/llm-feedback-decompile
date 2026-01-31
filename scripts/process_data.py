import os
import json
import sys
import hashlib
import multiprocessing as mp
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    DEDUP_DATA_DIR,
)
from src.utils import (
    compile_to_object,
    disasm_object,
    extract_asm_and_machine, 
)

SPLITS = ["train", "valid", "test"]
ARCHES = ["x86", "arm"]

def process_single_sample(c_code: str) -> list:
    """
    处理单个 C 函数代码样本，编译并提取汇编代码和机器码
    """
    entry = {
        "c_code": c_code,
        "compilations": {}
    }
    
    has_success = False
    created_workdirs = []
    try:
        for arch in ARCHES:
            success, _, o_path = compile_to_object(arch, c_code)
            if success and o_path:
                created_workdirs.append(Path(o_path).parent)
                disasm_result = disasm_object(arch, o_path)
                asm, machine_code = extract_asm_and_machine(arch, disasm_result)
                if asm and machine_code:
                    entry["compilations"][arch] = {
                        "asm": asm,
                        "machine_code": machine_code
                    }
                    has_success = True
        if has_success:
            return [entry]
        return []
    except Exception:
        return []
    finally:
        for workdir in created_workdirs:
            if workdir and workdir.exists():
                shutil.rmtree(workdir, ignore_errors=True)

def normalize_text(text: str) -> list:
    """
    对文本进行归一化处理(替换换行符、制表符和多个空格为单个空格，最后按空格分割为 tokens 列表)
    """
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

def lsh_deduplicate(records: list, field: str = "asm", num_perm: int = 64, bands: int = 8, shingle_size: int = 5) -> tuple:
    if not records:
        return [], []
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands")
    rows_per_band = num_perm // bands
    buckets = {}
    kept_indices = []
    kept_records = []
    for idx, rec in enumerate(records):
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

def deduplicate_jsonl(input_path: Path, output_path: Path, field: str = "asm", num_perm: int = 64, bands: int = 8, shingle_size: int = 5) -> tuple:
    input_path = Path(input_path)
    output_path = Path(output_path)
    try:
        dataset = load_dataset("json", data_files=str(input_path), split="train")
        records = [row for row in dataset]
    except Exception as e:
        records = []
    dedup_records, _ = lsh_deduplicate(
        records,
        field=field,
        num_perm=num_perm,
        bands=bands,
        shingle_size=shingle_size,
    )
    with output_path.open("w", encoding="utf-8") as f:
        for rec in dedup_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records), len(dedup_records)

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEDUP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in SPLITS:
        split_dir = RAW_DATA_DIR / split
        processed_out_path = PROCESSED_DATA_DIR / f"{split}_data.jsonl"
        deduplicated_out_path = DEDUP_DATA_DIR / f"{split}_data.jsonl"

        if processed_out_path.exists() and deduplicated_out_path.exists():
            print(f"\n{split} 的 data 文件已存在，跳过生成")
            continue

        if not processed_out_path.exists():
            jsonl_files = list(split_dir.glob("*.jsonl"))
            if split == "train":
                jsonl_files = jsonl_files[:6]
            total_files = len(jsonl_files)
            print(f"\n处理 {split} 数据集，共 {total_files} 个文件")

            tasks = []
            total_samples = 0
            for i, jsonl_file in enumerate(jsonl_files):
                print(f"处理文件({i+1}/{len(jsonl_files)}): {jsonl_file.name} ")
                with open(jsonl_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                        except:
                            continue
                        c_code = sample["text"]["func_def"].strip()
                        if c_code:
                            tasks.append(c_code)
                            total_samples += 1

            print(f"总共找到 {total_samples} 个 C 函数，开始多进程编译...")

            max_workers = max(1, os.cpu_count() - 1)
            print(f"使用 {max_workers} 个进程并行处理")

            results_generated = 0
            with open(processed_out_path, "w", encoding="utf-8") as out_f:
                if tasks:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        future_to_task = {executor.submit(process_single_sample, task): task for task in tasks}
                        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="编译进度"):
                            result_list = future.result()
                            for item in result_list:
                                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                                results_generated += 1

            print(f"\n{split} 处理完成！生成 {results_generated} 条数据")
            print(f"\n保存到: {processed_out_path}")
        else:
            print(f"\n{split} 的 processed_data 文件已存在，跳过生成: {processed_out_path}")

        if not deduplicated_out_path.exists():
            print(f"\n开始对 {split} 数据进行去重...")
            before_num, after_num = deduplicate_jsonl(
                input_path=processed_out_path,
                output_path=deduplicated_out_path,
                field="c_code",
                num_perm=64,
                bands=8,
                shingle_size=5,
            )
            print(f"{split} 去重完成: {before_num} -> {after_num}")
            print(f"去重后数据保存到: {deduplicated_out_path}")
        else:
            print(f"\n{split} 的 dedup_data 文件已存在，跳过生成: {deduplicated_out_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

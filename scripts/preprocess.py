import json
import subprocess
import tempfile
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
import sys

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, PROCESSED_DATA_DIR

base_dir = DATA_DIR
output_dir = PROCESSED_DATA_DIR
output_dir.mkdir(parents=True, exist_ok=True)

# SPLITS = ["train", "valid", "test"]
SPLITS = ["train"]

TARGETS = [
    ("x86", "O0"), ("x86", "O1"), ("x86", "O2"), ("x86", "O3"), 
    ("arm", "O0"), ("arm", "O1"), ("arm", "O2"), ("arm", "O3"), 
]

def extract_c_code(sample):
    """提取 C 代码"""
    try:
        return sample["text"]["func_def"].strip()
    except Exception:
        return None

def compile_to_machine_code(c_code: str, arch: str, opt_level: str):
    """编译 C 代码并提取机器码，返回 (arch, opt, machine_code)"""
    if arch == "x86":
        cc = "gcc"
        compile_args = ["-m64"]
        objdump_cmd = ["objdump", "-d"]
    elif arch == "arm":
        cc = "aarch64-linux-gnu-gcc"
        compile_args = []
        objdump_cmd = ["aarch64-linux-gnu-objdump", "-d"]
    else:
        return None

    c_path = None
    obj_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as c_file:
            c_file.write(c_code.encode("utf-8"))
            c_path = c_file.name

        with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as obj_file:
            obj_path = obj_file.name

        compile_cmd = [cc] + compile_args + [f"-{opt_level}", "-c", c_path, "-o", obj_path]
        r = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None

        objdump_cmd_full = objdump_cmd + [obj_path]
        r = subprocess.run(objdump_cmd_full, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None

        bytes_out = []
        for line in r.stdout.splitlines():
            if ':' not in line:
                continue
            _, rest = line.split(':', 1)
            for token in rest.strip().split():
                if all(c in "0123456789abcdefABCDEF" for c in token):
                    if len(token) == 2:  # x86 单字节
                        bytes_out.append(token.lower())
                    elif len(token) == 8:  # ARM 4 字节指令
                        bytes_out.extend([token.lower()[i:i+2] for i in range(0, 8, 2)])
                    else:
                        break
                else:
                    break

        machine_code = " ".join(bytes_out)
        if not machine_code:
            return None

        return (arch, opt_level, machine_code)

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        for path in [c_path, obj_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

def process_single_sample(args):
    """进程工作者函数：处理一个 C 代码样本，返回多个结果列表"""
    sample_json_line, c_code = args
    results = []
    for arch, opt in TARGETS:
        result = compile_to_machine_code(c_code, arch, opt)
        if result:
            arch_out, opt_out, mc = result
            results.append({
                "arch": arch_out,
                "opt": opt_out,
                "c_code": c_code,
                "machine_code": mc
            })
    return results

def main():
    for split in SPLITS:
        split_dir = base_dir / split
        jsonl_files = list(split_dir.glob("*.jsonl"))
        # 仅处理 train 数据集的前 6 个文件
        if split == "train":
            jsonl_files = jsonl_files[:6]
        total_files = len(jsonl_files)
        print(f"\n处理 {split} 数据集，共 {total_files} 个文件")
        
        # 使用配置中的 processed 目录
        out_path = output_dir / f"{split}_machine_to_c.jsonl"

        # 收集所有待处理的 (line, c_code)
        tasks = []
        total_samples = 0
        for i , jsonl_file in enumerate(jsonl_files):
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
                    c_code = extract_c_code(sample)
                    if c_code:
                        tasks.append((line, c_code))
                        total_samples += 1

        print(f"总共找到 {total_samples} 个有效 C 函数，开始多进程编译...")

        # 多进程处理
        max_workers = max(1, os.cpu_count() - 1)  # 留一个核给系统
        print(f"使用 {max_workers} 个进程并行处理")

        results_generated = 0
        with open(out_path, "w", encoding="utf-8") as out_f:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_task = {executor.submit(process_single_sample, task): task for task in tasks}

                # 实时进度条
                for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="编译进度"):
                    result_list = future.result()
                    for item in result_list:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        results_generated += 1

        print(f"{split} 处理完成！生成 {results_generated} 条 machine_to_c 数据")
        print(f"保存到: {out_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
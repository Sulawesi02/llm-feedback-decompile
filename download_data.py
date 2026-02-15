import json
import yaml
from datasets import load_dataset
from src.config import (
    RAW_DATA_DIR, 
    CONFIG_DIR,
)

def load_cfg():
    with open(CONFIG_DIR / "dataset.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def download_decompile_bench():
    print("\n=== 下载 Decompile-Bench (训练集) ===")
    output_file = RAW_DATA_DIR / "train_data.jsonl"
    
    if output_file.exists():
        print(f"文件已存在: {output_file}，跳过下载")
    else:
        try:
            cfg = load_cfg().get("decompile_bench", {})
            repo_id = cfg.get("repo_id")
            split = cfg.get("split")
            limit = cfg.get("max_samples")
            print(f"加载 {repo_id} 数据集...")
            dataset = load_dataset(repo_id, split=split, streaming=True)
            
            count = 0
            
            print(f"正在下载并保存 {limit} 条样本...")
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 1000 == 0:
                        print(f"已处理 {count} 条...", end="\r")
                    if count >= limit:
                        break
            
            print(f"Decompile-Bench 下载完成，共 {count} 条样本保存至: {output_file}")
            
        except Exception as e:
            print(f"下载 Decompile-Bench 失败: {e}")

def download_decompile_eval():
    print("\n=== 下载 Decompile-Eval (测试集) ===")
    output_file = RAW_DATA_DIR / "test_data.jsonl"
    
    if output_file.exists():
        print(f"文件已存在: {output_file}，跳过下载")
    else:
        try:
            cfg = load_cfg().get("decompile_eval", {})
            repo_id = cfg.get("repo_id")
            split = cfg.get("split")
            limit = cfg.get("max_samples")
            filt = cfg.get("filter", {})
            exclude_str = filt.get("exclude_func_dep_contains")
            require_test = bool(filt.get("require_test", True))
            opt_equals = filt.get("opt_equals", "O0")
            print(f"加载 {repo_id} 数据集...")
            dataset = load_dataset(repo_id, split=split, streaming=True)
            
            count = 0
            
            print(f"正在下载并保存 {limit} 条样本...")
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    func_dep = str(item.get("func_dep", "") or "")
                    if exclude_str and exclude_str in func_dep:
                        continue
                    if require_test and not item.get("test"):
                        continue
                    if opt_equals and item.get("opt") != opt_equals:
                        continue
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 1000 == 0:
                        print(f"已处理 {count} 条...", end="\r")
                    if count >= limit:
                        break
            
            print(f"Decompile-Eval 下载完成，共 {count} 条样本保存至: {output_file}")
            
        except Exception as e:
            print(f"下载 Decompile-Eval 失败: {e}")

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_decompile_bench()
    download_decompile_eval()
    print("\n数据集下载完成！")

if __name__ == "__main__":
    main()

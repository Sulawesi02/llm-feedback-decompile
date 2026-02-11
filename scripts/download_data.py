import json
from pathlib import Path
import sys
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_DIR

def download_decompile_bench():
    print("\n=== 下载 Decompile-Bench (训练集) ===")
    output_file = RAW_DATA_DIR / "train_data.jsonl"
    
    if output_file.exists():
        print(f"文件已存在: {output_file}，跳过下载")
    else:
        try:
            print("加载 LLM4Binary/decompile-bench 数据集...")
            dataset = load_dataset("LLM4Binary/decompile-bench", split="train", streaming=True)
            
            # 尝试加载前 20000 条作为训练数据
            count = 0
            limit = 20000 
            
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
            print("加载 LLM4Binary/decompile-eval 数据集...")
            dataset = load_dataset("LLM4Binary/decompile-eval", split="train", streaming=True)
            
            # 尝试加载前 2000 条作为测试数据
            count = 0
            limit = 2000 
            
            print(f"正在下载并保存 {limit} 条样本...")
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    # func_dep不能有”#define”，且 test 不为空，且 opt = O0
                    if "#define" in str(item.get("func_dep", "")) or not item.get("test") or item.get("opt") != "O0":
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

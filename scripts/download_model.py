import os
import time
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# 添加项目根目录到 sys.path 以便导入 config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BASE_MODEL_DIR_PATH

# 设置 HF 镜像源 (如果没有设置的话)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAMES = [
    "Qwen2.5-Coder-1.5B-Instruct",
    "Qwen2.5-Coder-3B-Instruct",
    "Qwen2.5-Coder-7B-Instruct",
]

def download_base_model(repo_id: str, local_dir: Path, max_retries=10):
    """
    下载基座模型，支持自动重试和断点续传
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(max_retries):
        try:
            print(f"\n[{i+1}/{max_retries}] 开始下载/验证模型文件...")
            
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"模型下载完成: {path}")
            return path
        except Exception as e:
            print(f"模型下载失败: {e}")
            if i < max_retries - 1:
                wait_time = 5 * (i + 1)  # 递增等待时间
                print(f"{wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"模型下载失败，已重试 {max_retries} 次") from e

def main():    
    for model_name in MODEL_NAMES:
        model_path = BASE_MODEL_DIR_PATH / model_name
        
        if model_path.exists() and (model_path / "config.json").exists():
            print(f"{model_name} 已存在，跳过下载。")
            continue
            
        try:
            print(f"开始下载 {model_name}")
            download_base_model(model_name, model_path)
        except Exception as e:
            print(f"{model_name} 下载失败: {e} !!")
            continue
            
    print("所有下载任务处理完成！")

if __name__ == "__main__":
    main()

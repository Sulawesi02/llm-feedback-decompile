import os
import time
import sys
from pathlib import Path

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_NAME, BASE_MODEL_DIR

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
    if not MODEL_NAME:
        print("错误: 模型名称未配置")
        return
    if not BASE_MODEL_DIR:
        print("错误: 基座模型目录未配置")
        return
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    base_model_path.mkdir(parents=True, exist_ok=True)
    
    if base_model_path.exists() and (base_model_path / "config.json").exists():
        print(f"{MODEL_NAME} 已存在，跳过下载。")
        return
    
    try:
        download_base_model(MODEL_NAME, base_model_path)
    except Exception as e:
        print(f"{MODEL_NAME} 下载失败: {e} !!")

if __name__ == "__main__":
    main()

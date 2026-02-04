import os
import time
import sys
import tarfile
import subprocess
from pathlib import Path

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_DIR

# 数据集配置
DATASET_REPO_ID = "jordiae/exebench"

SPLITS = ["train", "valid", "test"]

def download_dataset(local_dir: Path, max_retries=10):
    """
    下载 ExeBench 数据集，支持自动重试
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"准备下载 ExeBench 数据集到: {local_dir}")
    
    for i in range(max_retries):
        try:
            print(f"\n[{i+1}/{max_retries}] 开始下载数据集文件...")
            # allow_patterns 只下载我们需要的那 6 个文件
            path = snapshot_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=[
                    "train_synth_simple_io.tar.gz",
                    "train_real_simple_io.tar.gz",
                    "valid_synth.tar.gz",
                    "valid_real.tar.gz",
                    "test_synth.tar.gz",
                    "test_real.tar.gz",
                ]
            )
            print(f"数据集下载完成: {path}")
            return path
        except Exception as e:
            print(f"下载失败: {e}")
            if i < max_retries - 1:
                wait_time = 5 * (i + 1)
                print(f"{wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"数据集下载失败，已重试 {max_retries} 次") from e

def unzip_dataset(base_dir: Path):
    """
    解压 ExeBench 数据集文件
    """
    print("\n=== 开始解压 ExeBench 数据集 ===")
    os.chdir(base_dir)

    # 1. 创建目录结构
    print("1. 创建目录结构 (train, valid, test)...")
    for d in SPLITS:
        (base_dir / d).mkdir(parents=True, exist_ok=True)

    # 2. 解压 .tar.gz 文件
    print("2. 处理 .tar.gz 文件...")
    tar_files = {
        "train": ["train_synth_simple_io.tar.gz", "train_real_simple_io.tar.gz"],
        "valid": ["valid_synth.tar.gz", "valid_real.tar.gz"],
        "test": ["test_synth.tar.gz", "test_real.tar.gz"]
    }

    for category, files in tar_files.items():
        for file_name in files:
            file_path = base_dir / file_name
            if file_path.exists():
                print(f"  正在解压 {file_name} 到 {category}/ ...")
                try:
                    with tarfile.open(file_path, "r:gz") as tar:
                        for member in tar.getmembers():
                            if '/' in member.name:
                                member.name = member.name.split('/', 1)[1]
                                if member.name: # 避免解压根目录本身
                                    tar.extract(member, path=base_dir / category)
                    file_path.unlink() # 解压后删除
                except Exception as e:
                    print(f"  解压 {file_name} 失败: {e}")
            else:
                print(f"  跳过 {file_name} (文件不存在)")

    # 3. 解压 .zst 文件
    print("3. 处理 .jsonl.zst 文件...")
    zst_files = list(base_dir.glob("**/*.jsonl.zst"))
    if zst_files:
        # 检查是否安装了 zstd
        try:
            subprocess.run(["zstd", "--version"], capture_output=True, check=True)
            has_zstd = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  错误: 系统未安装 zstd 命令，无法解压 .zst 文件。")
            print("  请先运行 'apt-get install zstd' 或手动解压。")
            has_zstd = False

        if has_zstd:
            for file_path in zst_files:
                output_path = file_path.with_suffix("") # 去掉 .zst
                print(f"  正在解压 {file_path.name} ...")
                try:
                    subprocess.run(["zstd", "-d", str(file_path), "-o", str(output_path)], check=True)
                    file_path.unlink() # 解压后删除
                except Exception as e:
                    print(f"  解压 {file_path.name} 失败: {e}")
    else:
        print("  未发现 .jsonl.zst 文件。")

    print("=== 解压完成 ===")

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 下载
        download_dataset(RAW_DATA_DIR)
        
        # 2. 解压
        unzip_dataset(RAW_DATA_DIR)
        
        print("\nExeBench 数据集准备就绪！")
    except Exception as e:
        print(f"\n错误: {e}")

if __name__ == "__main__":
    main()

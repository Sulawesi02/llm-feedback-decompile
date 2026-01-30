import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_DATA, 
    DPO_DATA_DIR, 
    MODEL_NAME,
    BASE_MODEL_DIR, 
    VERSIONS,
    OFFLOAD_ROOT,
)
from src.utils import extract_compilation_data, ModelRunning
from src.prompts import construct_bad_code_prompt, construct_train_prompt

DPO_DATA_RATIO = 0.05 # DPO 数据生成采样比例

def main():
    if not TRAIN_DATA.exists():
        print(f"错误: 训练集不存在: {TRAIN_DATA}")
        return
    DPO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_NAME:
        print("错误: 模型名称未配置")
        return
    if not BASE_MODEL_DIR.exists():
        print(f"错误: 基础模型目录不存在: {BASE_MODEL_DIR}")
        return
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    if not base_model_path.exists():
        print(f"错误: 基础模型不存在: {base_model_path}")
        return
    if not VERSIONS:
        print("错误: 版本未配置")
        return
    
    offload_dir = OFFLOAD_ROOT / "process_dpo"
    offload_dir.mkdir(parents=True, exist_ok=True)
    
    model_running = ModelRunning(
        base_model_path=str(base_model_path),
        offload_folder=str(offload_dir),
        offload_buffers=False
    )

    print("加载数据集...")
    train_ds = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
    subset_size = int(len(train_ds) * DPO_DATA_RATIO)
    final_train = train_ds.select(range(subset_size))
    
    dpo_data = []
    
    print(f"开始生成 DPO 数据 (Rejected Samples)... 样本数: {len(final_train)}")
    
    all_tasks = []
    for item in final_train:
        for arch, asm, c_code, _ in extract_compilation_data(item):
            all_tasks.append((arch, asm, c_code))
    
    for arch, asm, c_code in tqdm(all_tasks, desc=f"生成 DPO 数据"):
        # 1. 构造用于 DPO 数据集的 Prompt (Assembly -> C)
        full_messages = construct_train_prompt(arch, asm, "")
        prompt_messages = full_messages[:2] # System + User
        prompt_text = model_running.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        # 2. 构造用于生成 Rejected Sample 的 Prompt (C -> Bad C)
        bad_code_prompt = construct_bad_code_prompt(c_code)
        
        rejected_code = model_running.generate(
            messages=bad_code_prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.9,
            top_p=0.9
        )
        
        entry = {
            "prompt": prompt_text, 
            "chosen": c_code,
            "rejected": rejected_code
        }
        dpo_data.append(entry)
        
    # 保存 - 不区分版本，直接保存到 dpo_data 目录下的 dpo_data.jsonl
    output_dir = DPO_DATA_DIR / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dpo_data.jsonl"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
    print(f"保存 {len(dpo_data)} 条数据到 {output_file}")
    
    # 释放资源
    model_running.unload()
    del model_running
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

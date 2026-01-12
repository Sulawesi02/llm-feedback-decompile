import os
import time
import torch
import json
import gc
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BASE_MODEL_NAME, BASE_MODEL_PATH, LORA_CHECKPOINTS_DIR, MERGED_MODEL_DIR, TRAIN_DATA_PATH, VALID_DATA_PATH

# 设置 HF 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType
from datasets import load_dataset

DATA_FILES = {
    "train": str(TRAIN_DATA_PATH),
    "valid": str(VALID_DATA_PATH)
}

def download_model_with_retry(repo_id, local_dir, max_retries=10):
    for i in range(max_retries):
        try:
            print(f"[{i+1}/{max_retries}] 正在下载模型: {repo_id}")
            
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"模型已准备就绪: {path}")
            return path
        except Exception as e:
            print(f"下载失败: {e}")
            if i < max_retries - 1:
                print("10秒后重试...")
                time.sleep(10)
            else:
                raise RuntimeError(f"模型下载失败，已重试 {max_retries} 次") from e

def format_and_tokenize(examples, tokenizer, max_length=256):
    """
    将数据集格式化为模型输入并进行 Tokenization
    """
    inputs = []
    targets = []
    
    for arch, opt, machine_code, c_code in zip(
        examples['arch'], examples['opt'], examples['machine_code'], examples['c_code']
    ):
        # 构建 Prompt
        prompt = f"""你是一个专业的二进制反编译专家。
请把下面这段 {arch} {opt} 机器码反编译成可读的 C 语言函数：

{machine_code}

输出只包含 C 代码，不要解释，不要添加额外文字。

C 代码：
"""
        # 输入部分 = Prompt + Answer + EOS
        full_text = prompt + c_code + tokenizer.eos_token
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        inputs.append(tokenized["input_ids"])
        targets.append(tokenized["input_ids"])

    return {
        "input_ids": inputs,
        "attention_mask": [ [1]*len(x) for x in inputs ],
        "labels": targets
    }

def train_version(version_name, data_ratio):
    print(f"\n{'='*20} 开始训练版本: {version_name} (数据比例: {data_ratio}) {'='*20}")
    
    # 转换为 str 避免 path 对象拼接问题（如果库不支持）
    lora_version_dir = str(LORA_CHECKPOINTS_DIR / version_name)
    merged_model_version_dir = str(MERGED_MODEL_DIR / version_name)
    base_model_path_str = str(BASE_MODEL_PATH)
    
    os.makedirs(lora_version_dir, exist_ok=True)
    os.makedirs(merged_model_version_dir, exist_ok=True)
    
    # 检查已生成的模型是否完整
    lora_exists = len(os.listdir(lora_version_dir)) > 0
    merged_exists = len(os.listdir(merged_model_version_dir)) > 0

    if not lora_exists:
        print(f"版本 {version_name} 的 LoRA 不存在，开始训练...")
        
        print("定义量化配置...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"加载 Tokenizer: {base_model_path_str}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path_str, trust_remote_code=True)

        print(f"加载模型: {base_model_path_str}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path_str,
            quantization_config=quant_config,
            device_map={"": 0}, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2", 
        )

        # 准备模型用于 k-bit 训练
        model = prepare_model_for_kbit_training(model)

        print("配置 LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        print("加载数据...")
        try:
            data = load_dataset("json", data_files=DATA_FILES)
            full_train = data["train"].shuffle(seed=42).select(range(1000))
            target_size = int(1000 * data_ratio)
            data["train"] = full_train.select(range(target_size))
            
            if "valid" in data:
                data["valid"] = data["valid"].shuffle(seed=42).select(range(200))
            
            print(f"当前训练集大小: {len(data['train'])} (Target: {target_size})")

            print("处理数据 (Tokenization)...")
            remove_columns = data["train"].column_names
            tokenized_dataset = data.map(
                lambda x: format_and_tokenize(x, tokenizer),
                batched=True,
                remove_columns=remove_columns
            )
        except Exception as e:
            print(f"警告: 数据加载或处理失败: {e}")
            return

        training_args = TrainingArguments(
            output_dir=lora_version_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=3e-4,
            num_train_epochs=3,
            bf16=True,
            fp16=False,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            logging_steps=20,
            eval_strategy="steps", 
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            report_to=[], 
            disable_tqdm=False,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["valid"] if "valid" in tokenized_dataset else None,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )

        print(f"开始训练版本 {version_name}...")
        trainer.train()

        print(f"保存 LoRA 适配器到 {lora_version_dir}")
        model.save_pretrained(lora_version_dir)
        tokenizer.save_pretrained(lora_version_dir)
        
        # 清理训练资源
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    # 2. 如果 Merged 模型不存在，则进行合并
    if not merged_exists:
        print(f"版本 {version_name} 的 LoRA 已准备就绪，开始合并模型...")

        print(f"开始合并模型并保存到 {merged_model_version_dir} ...")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path_str, trust_remote_code=True)

        # 重新加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path_str,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 加载 LoRA
        model_to_merge = PeftModel.from_pretrained(base_model, lora_version_dir)
        
        # 合并
        merged_model = model_to_merge.merge_and_unload()
        
        # 保存
        merged_model.save_pretrained(merged_model_version_dir)
        tokenizer.save_pretrained(merged_model_version_dir)
        
        print(f"版本 {version_name} 合并完成！")
        
        # 清理合并后的模型
        del base_model, model_to_merge, merged_model
        torch.cuda.empty_cache()
        gc.collect()
    
    elif lora_exists and merged_exists:
        print(f"版本 {version_name} 的 LoRA 和合并模型均已存在且完整，跳过训练。")
        return
    
def main():
    # 确保目录存在
    BASE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    LORA_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    base_exists = len(os.listdir(BASE_MODEL_PATH)) > 0
    
    print("开始准备基座模型...")
    if base_exists:
        print(f"模型已存在: {BASE_MODEL_PATH}")
    else:
        print(f"模型不存在, 开始下载模型...")
        try:
            download_model_with_retry(BASE_MODEL_NAME, BASE_MODEL_PATH)
        except Exception as e:
            print(f"下载模型失败: {e}")
            return

    versions = [
        ("v1", 0.10),
        ("v2", 0.25),
        # ("v3", 0.50),
        # ("v4", 1.00)
    ]
    
    for ver, ratio in versions:
        try:
            train_version(ver, ratio)
        except Exception as e:
            print(f"训练版本 {ver} 失败: {e}")
            break

    print("\n训练结束！")

if __name__ == "__main__":
    main()

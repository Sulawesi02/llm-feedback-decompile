import os
import torch
import gc
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BASE_MODEL_DIR_PATH, LORA_CHECKPOINTS_DIR_PATH, MERGED_MODEL_DIR_PATH, TRAIN_DATA_FILE_PATH, VALID_DATA_FILE_PATH
from src.utils import load_model_utils, load_jsonl

# 设置 HF 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType
from datasets import Dataset, DatasetDict

MODEL_NAMES = [
    # "Qwen2.5-Coder-1.5B-Instruct",
    # "Qwen2.5-Coder-3B-Instruct",
    "Qwen2.5-Coder-7B-Instruct",
]

VERSIONS = [
    ("v1", 0.01),
    ("v2", 0.02),
    ("v3", 0.05),
    # ("v4", 1.00)
]

def format_and_tokenize(examples, tokenizer, max_length=256):
    """ 格式化并 tokenize 输入样本 """
    inputs = []
    targets = []
    for c_code, compilations_data in zip(examples["c_code"], examples["compilations"]):
        for arch, arch_data in compilations_data.items():
            for opt, opt_data in arch_data.items():
                asm_text = opt_data.get("asm")
                
                prompt = (
                    f"架构: {arch}\n"
                    f"优化级别: {opt}\n\n"
                    "请将以下汇编代码反编译成等效的 C 函数，要求：\n"
                    "1. 严格保持语义正确；\n"
                    "2. 尽量使用清晰易读的 C 语言写法（如数组下标和规范的控制流结构）；\n"
                    "3. 只输出一个完整的 C 函数定义，不要包含任何解释性文字或 Markdown 代码块标记。\n\n"
                    f"{asm_text}\n\n"
                )
                text = prompt + (c_code or "")
                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None,
                )
                inputs.append(tokenized["input_ids"])
                targets.append(tokenized["input_ids"])

    return {
        "input_ids": inputs,
        "attention_mask": [[1] * len(x) for x in inputs],
        "labels": targets,
    }

def train_model(model_name, version, ratio):
    base_model_path = BASE_MODEL_DIR_PATH / model_name
    
    if not base_model_path.exists():
        print(f"基座模型不存在: {base_model_path}，跳过 {model_name} 的 {version}")
        return
    
    print(f"\n{'='*20} 开始训练模型: {model_name}, 版本: {version} (数据比例: {ratio}) {'='*20}")
    
    lora_path = LORA_CHECKPOINTS_DIR_PATH / model_name / version
    merge_model_path = MERGED_MODEL_DIR_PATH / model_name / version
    
    lora_path.mkdir(parents=True, exist_ok=True)
    merge_model_path.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在完整的 LoRA 和合并模型
    lora_exists = lora_path.exists() and (lora_path / "adapter_config.json").exists()
    merged_exists = merge_model_path.exists() and (merge_model_path / "config.json").exists()

    if lora_exists and merged_exists:
        print(f"{model_name} 的 {version} 版本的 LoRA 和合并模型均已存在，跳过训练。")
        return
    
    if not lora_exists:
        print(f"{model_name} 的 {version} 版本的 LoRA 不存在，开始训练...")
        
        # 加载量化模型和分词器
        model, tokenizer = load_model_utils(str(base_model_path))

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
            train_raw = load_jsonl(TRAIN_DATA_FILE_PATH)
            valid_raw = load_jsonl(VALID_DATA_FILE_PATH)
            
            print(f"原始训练数据: {len(train_raw)} 条, 验证数据: {len(valid_raw)} 条")
            
            train_ds = Dataset.from_list(train_raw).shuffle(seed=42)
            valid_ds = Dataset.from_list(valid_raw).shuffle(seed=42)
            
            train_target_size = max(1, int(len(train_ds) * ratio))
            valid_target_size = max(1, int(len(valid_ds) * ratio)) if len(valid_ds) > 0 else 0
            
            final_train = train_ds.select(range(train_target_size))
            final_valid = valid_ds.select(range(valid_target_size)) if valid_target_size > 0 else valid_ds
            
            data = DatasetDict({
                "train": final_train,
                "valid": final_valid
            })
            
            print(
                f"当前训练集大小: {len(data['train'])} (Target: {train_target_size}), "
                f"验证集大小: {len(data['valid'])} (Target: {valid_target_size})"
            )

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
            output_dir=lora_path,
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

        print(f"开始训练模型 {model_name} 版本 {version}...")
        trainer.train()

        print(f"保存 LoRA 适配器到 {lora_path}")
        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)
        
        # 清理训练资源
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    # 如果 Merged 模型不存在，则进行合并
    if not merged_exists:
        print(f"{model_name} 的 {version} 版本的合并模型不存在，开始合并...")
        
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)

        # 重新加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 加载 LoRA
        model_to_merge = PeftModel.from_pretrained(base_model, lora_path)
        
        # 合并
        merged_model = model_to_merge.merge_and_unload()
        
        # 保存
        merged_model.save_pretrained(merge_model_path)
        tokenizer.save_pretrained(merge_model_path)
        
        print(f"{model_name} 的 {version} 版本合并完成！")
        
        # 清理合并后的模型
        del base_model, model_to_merge, merged_model
        torch.cuda.empty_cache()
        gc.collect()
    
def main():
    LORA_CHECKPOINTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    MERGED_MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
    
    for model_name in MODEL_NAMES:
        for version, ratio in VERSIONS:
            try:
                train_model(model_name, version, ratio)
            except Exception as e:
                print(f"模型 {model_name} 版本 {version} 训练失败: {e}")
                break

    print("\n训练结束！")

if __name__ == "__main__":
    main()

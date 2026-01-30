import sys
import torch
import gc
from pathlib import Path
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_DATA, 
    VALID_DATA,
    MODEL_NAME,
    BASE_MODEL_DIR, 
    SFT_ADAPTER_DIR, 
    VERSIONS,
)
from src.utils import (
    ModelRunning, 
    extract_compilation_data,
)
from src.prompts import construct_train_prompt

def format_data(examples, tokenizer):
    """ 将样本转换为 SFTTrainer 需要的格式 (text list) """
    texts = []
    try:
        batch_size = len(examples["c_code"])
        
        for i in range(batch_size):
            item = {
                "c_code": examples["c_code"][i],
                "compilations": examples["compilations"][i]
            }
            
            for arch, asm, c_code, _ in extract_compilation_data(item):
                messages = construct_train_prompt(arch, asm, c_code)
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                texts.append(text)
        
        return {"text": texts}
        
    except Exception as e:
        print(f"格式化过程中出错: {e}")
        return {"text": []}

def train_sft(base_model_path, ratio, sft_adapter_path):
    print("加载基座模型和分词器...")
    model_running = ModelRunning(base_model_path=str(base_model_path))
    model = model_running.model
    tokenizer = model_running.tokenizer

    # 准备模型用于 k-bit 训练
    model = prepare_model_for_kbit_training(model)

    print("配置 LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )

    training_args = SFTConfig(
        output_dir=str(sft_adapter_path),
        per_device_train_batch_size=1,      # 每个设备的 batch size
        gradient_accumulation_steps=8,      # 梯度累积步数，模拟更大的 batch size
        learning_rate=3e-4,                 # 学习率
        num_train_epochs=3,                 # 训练轮数
        bf16=True,                          # 使用 bfloat16 精度 (推荐 Ampere+ GPU)
        fp16=False,
        optim="paged_adamw_8bit",           # 使用 8-bit 优化器节省显存
        gradient_checkpointing=True,        # 梯度检查点，节省显存但通过重计算换取时间
        logging_steps=20,                   # 每隔多少步打印日志
        eval_strategy="steps",              # 评估策略：按步数
        eval_steps=100,                     # 每隔多少步评估一次
        save_strategy="steps",              # 保存策略：按步数
        save_steps=100,                     # 每隔多少步保存一次
        load_best_model_at_end=True,        # 训练结束加载最好的模型
        metric_for_best_model="loss",       # 根据 loss 选择最好的模型
        report_to=[],                       # 不上报到 wandb 等平台
        disable_tqdm=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,        # SFT 必须保留 dataset_text_field 列，否则无法计算 loss
        
        # SFT 特有参数
        dataset_text_field="text",          # 数据集中包含训练文本的列名
        max_seq_length=2048,                # 单个样本的最大 Token 长度 (Prompt + Response)
        packing=False,                      # 是否开启 packing (将多个短样本拼接，提高训练效率，但可能影响收敛)
    )
    
    print("加载数据集...")
    raw_train = load_dataset("json", data_files=str(TRAIN_DATA), split="train")
    raw_valid = load_dataset("json", data_files=str(VALID_DATA), split="train")
    print(f"原始训练数据: {len(raw_train)} 条, 验证数据: {len(raw_valid)} 条")
    
    sampled_train = raw_train.shuffle(seed=42).select(range(int(len(raw_train) * ratio)))
    sampled_valid = raw_valid.shuffle(seed=42).select(range(int(len(raw_valid) * ratio)))
    print(f"采样后训练数据: {len(sampled_train)} 条, 验证数据: {len(sampled_valid)} 条")
    
    data = DatasetDict({
        "train": sampled_train,
        "valid": sampled_valid
    })
    
    print("数据预处理...")
    remove_columns = data["train"].column_names
    processed_dataset = data.map(
        lambda x: format_data(x, tokenizer),
        batched=True,
        remove_columns=remove_columns,
        desc="Formatting dataset"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["valid"] if "valid" in processed_dataset else None,
        tokenizer=tokenizer,
        peft_config=lora_config,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # SFTTrainer 默认有 collator，但也可以指定
    )

    print(f"开始 SFT 训练...")
    trainer.train()

    print(f"保存 LoRA 适配器到 {sft_adapter_path}")
    model.save_pretrained(sft_adapter_path)
    tokenizer.save_pretrained(sft_adapter_path)
    
    # 清理训练资源
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
def main():
    if not TRAIN_DATA.exists():
        print(f"错误: 训练集不存在: {TRAIN_DATA}")
        return
    if not VALID_DATA.exists():
        print(f"错误: 验证集不存在: {VALID_DATA}")
        return
    if not MODEL_NAME:
        print("错误: 模型名称未配置")
        return
    if not BASE_MODEL_DIR:
        print("错误: 基座模型目录未配置")
        return
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    if not base_model_path.exists():
        print(f"错误: 基座模型不存在: {base_model_path}")
        return
    SFT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    if not VERSIONS:
        print(f"错误: 版本号未配置")
        return
    
    for version, ratio in VERSIONS:
        sft_adapter_path = SFT_ADAPTER_DIR / version
        sft_adapter_exists = sft_adapter_path.exists() and (sft_adapter_path / "adapter_config.json").exists()
        
        if sft_adapter_exists:
            print(f"({version} 版本) SFT 适配器已存在，跳过训练")
            continue
        else:
            sft_adapter_path.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*20} 开始训练 ({version} 版本) SFT 适配器 (数据比例: {ratio}) {'='*20}")
            try:
                train_sft(base_model_path, ratio, sft_adapter_path)
            except Exception as e:
                print(f"({version} 版本) SFT 适配器训练失败: {e}")
                break

if __name__ == "__main__":
    main()

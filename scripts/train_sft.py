import sys
import torch
import gc
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from peft import prepare_model_for_kbit_training, get_peft_model

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SFT_DATA_DIR,
    MODEL_DIR, 
    MODEL_NAME,
    SFT_DIR, 
    VERSIONS,
    QUANT_CONFIG,
    LORA_CONFIG,
)

_tokenizer = None

def get_tokenizer(base_model_path: Path) -> AutoTokenizer:
    """获取或创建tokenizer（单例模式）"""
    global _tokenizer
    
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "right"
        _tokenizer.truncation_side = "right"
    
    return _tokenizer

def format_sft_data(example):
    """
    格式化 SFT 数据项
    """
    messages = [
        {"role": "system", "content": example.get("instruction", "")},
        {"role": "user", "content": example.get("response", "")},
    ]
    text = _tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False,
        return_dict=False,
    )
    return {"text": text}

def train_sft(base_model_path, sft_adapter_path, ratio):
    print("加载分词器...")
    _tokenizer = get_tokenizer(base_model_path)
    
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        quantization_config=QUANT_CONFIG,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LORA_CONFIG) # 绑定 LoRA
    model.print_trainable_parameters() # 打印可训练参数
    
    sft_args = SFTConfig(
        output_dir=str(sft_adapter_path),
        per_device_train_batch_size=1,      # 每个设备的训练 batch size
        per_device_eval_batch_size=1,       # 每个设备的验证 batch size
        gradient_accumulation_steps=8,      # 梯度累积步数，模拟更大的 batch size
        learning_rate=3e-4,                 # 学习率
        num_train_epochs=3,                 # 训练轮数
        logging_steps=10,                   # 打印日志频率
        save_steps=100,                     # 保存模型检查点频率
        eval_steps=100,                     # 评估模型频率
        eval_strategy="steps",              # 评估策略
        load_best_model_at_end=True,        # 训练结束后加载最优模型
        metric_for_best_model="eval_loss",  # 用于选择最优模型的指标
        fp16=torch.cuda.is_available(),     # 使用 float16 精度
        warmup_ratio=0.1,                   # 预热比例
        weight_decay=0.01,                  # 权重衰减
        report_to=["tensorboard"],          # 启用 TensorBoard 记录
        gradient_checkpointing=True,        # 启用梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},# 设置非重入模式
        optim="adamw_8bit",                 # 使用8位优化器
        group_by_length=False,              # 按长度分组
        dataset_text_field="text",          # 数据集文本字段
        max_seq_length=512,                 # 最大序列长度
        packing=False,                      # 是否使用打包
    )
    
    print(f"加载数据集...")
    train_data = load_dataset("json", data_files=str(SFT_DATA_DIR / "train_data.jsonl"), split="train")
    valid_data = load_dataset("json", data_files=str(SFT_DATA_DIR / "valid_data.jsonl"), split="train")
    print(f"原始训练数据: {len(train_data)} 条")
    print(f"原始验证数据: {len(valid_data)} 条")
    
    print(f"采样数据集...")
    sampled_train = train_data.shuffle(seed=42).select(range(int(len(train_data) * ratio)))
    sampled_valid = valid_data.shuffle(seed=42).select(range(int(len(valid_data) * ratio)))
    print(f"采样训练数据: {len(sampled_train)} 条")
    print(f"采样验证数据: {len(sampled_valid)} 条")
    
    print(f"格式化数据集...")
    sampled_train = sampled_train.map(format_sft_data)
    sampled_valid = sampled_valid.map(format_sft_data)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=sampled_train,
        eval_dataset=sampled_valid,
        peft_config=LORA_CONFIG,
        tokenizer=_tokenizer,
        args=sft_args,
    )

    print(f"开始 SFT 训练...")
    last_checkpoint = get_last_checkpoint(str(sft_adapter_path))
    if last_checkpoint is not None:
        print(f"发现 checkpoint: {last_checkpoint}, 断点续训")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f"未发现有效 checkpoint, 从头训练")
        trainer.train()

    print(f"保存 LoRA 适配器...")
    trainer.save_model(sft_adapter_path)
    
    # 清理资源
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    base_model_path = MODEL_DIR / MODEL_NAME
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    
    for version, ratio in VERSIONS:
        sft_adapter_path = SFT_DIR / version
        sft_adapter_exists = sft_adapter_path.exists() and (sft_adapter_path / "adapter_config.json").exists()
        
        if sft_adapter_exists:
            print(f"({version} 版本) SFT 适配器已存在，跳过训练")
            continue
        else:
            sft_adapter_path.mkdir(parents=True, exist_ok=True)
            print(f"{'='*20} 开始训练 ({version} 版本) SFT 适配器 (数据比例: {ratio}) {'='*20}")
            try:
                train_sft(base_model_path, sft_adapter_path, ratio)
            except Exception as e:
                print(f"({version} 版本) SFT 适配器训练失败: {e}")
                break

if __name__ == "__main__":
    main()
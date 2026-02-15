import torch
import gc
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import yaml
from transformers.trainer_utils import get_last_checkpoint

from src.config import (
    PROCESSED_DATA_DIR,
    CONFIG_DIR,
    MODEL_DIR, 
    MODEL_NAME,
    SFT_DIR,
    MAX_PROMPT_TOKENS,
)
def load_sft_cfg():
    with open(CONFIG_DIR / "sft.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_sft(base_model_path, SFT_DIR):
    print(f"{'='*20} 开始训练 SFT 适配器 {'='*20}")
    cfg = load_sft_cfg()
    tcfg = cfg.get("training", {}) if cfg else {}
    lcfg = cfg.get("lora", {}) if cfg else {}
    ctcfg = cfg.get("chat_template", {}) if cfg else {}
    max_seq_len = int(tcfg.get("max_seq_length", MAX_PROMPT_TOKENS))
    
    
    print("加载分词器和基座模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(base_model_path),
        max_seq_length = max_seq_len,
        dtype = None,
        load_in_4bit = True,
        device_map="auto",
    )

    print("设置 Qwen2.5 聊天模板...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = ctcfg.get("name", "qwen2.5"),
    )
    
    print("配置 PEFT 模型...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = int(lcfg.get("r", 8)),
        target_modules = lcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_alpha = int(lcfg.get("lora_alpha", 16)),
        lora_dropout = float(lcfg.get("lora_dropout", 0)),
        bias = str(lcfg.get("bias", "none")),
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
        max_seq_length = max_seq_len,
    )

    print("加载数据集...")
    train_data = load_dataset("json", data_files=str(PROCESSED_DATA_DIR / "train_data.jsonl"), split="train")
    valid_data = load_dataset("json", data_files=str(PROCESSED_DATA_DIR / "valid_data.jsonl"), split="train")
    print(f"训练样本数: {len(train_data)}")
    print(f"验证样本数: {len(valid_data)}")

    # 格式化函数
    def format_qwen_chat(example):
        messages = [
            {"role": "system", "content": "你是一名专业的 C 语言反编译专家，精通 x86-64 汇编。请根据给定的汇编代码，生成语义上完全等价的 C 函数实现。"},
            {"role": "user", "content": f"请根据以下 x86-64 汇编代码，生成等价的 C 函数实现：\n\n```assembly\n{example['asm']}\n```"},
            {"role": "assistant", "content": f"```c\n{example['func']}\n```"},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("格式化数据集...")
    formatted_train = train_data.map(format_qwen_chat, remove_columns=train_data.column_names)
    formatted_valid = valid_data.map(format_qwen_chat, remove_columns=valid_data.column_names)

    print("配置 SFT 训练参数...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_train,
        eval_dataset=formatted_valid,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=False,
        args=SFTConfig(
            output_dir=str(SFT_DIR),
            per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
            learning_rate=float(tcfg.get("learning_rate", 3e-4)),
            lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
            warmup_ratio=float(tcfg.get("warmup_ratio", 0.1)),
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
            num_train_epochs=int(tcfg.get("num_train_epochs", 3)),
            fp16=bool(tcfg.get("fp16", False)),
            bf16=bool(tcfg.get("bf16", True)),
            optim=str(tcfg.get("optim", "adamw_8bit")),
            logging_steps=int(tcfg.get("logging_steps", 10)),
            save_strategy=str(tcfg.get("save_strategy", "steps")),
            save_steps=int(tcfg.get("save_steps", 100)),
            eval_strategy=str(tcfg.get("eval_strategy", "steps")),
            eval_steps=int(tcfg.get("eval_steps", 100)),
            save_total_limit=int(tcfg.get("save_total_limit", 3)),
            load_best_model_at_end=bool(tcfg.get("load_best_model_at_end", True)),
            metric_for_best_model=str(tcfg.get("metric_for_best_model", "eval_loss")),
            greater_is_better=bool(tcfg.get("greater_is_better", False)),
            dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 4)),
            remove_unused_columns=bool(tcfg.get("remove_unused_columns", False)),
            max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
            gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
            report_to="tensorboard",
        ),
    )
    print(f"开始 SFT 训练...")
    
    # 检查是否有 checkpoint 用于恢复训练
    last_checkpoint = get_last_checkpoint(str(SFT_DIR))
    if last_checkpoint is not None:
        print(f"发现已有 SFT checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    print(f"保存 LoRA 适配器...")
    trainer.save_model(SFT_DIR)
    
    # 清理资源
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    base_model_path = MODEL_DIR / MODEL_NAME
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        train_sft(base_model_path, SFT_DIR)
    except Exception as e:
        print(f"SFT 适配器训练失败: {e}")

if __name__ == "__main__":
    main()

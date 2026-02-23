import torch
import gc
from unsloth import FastLanguageModel
from peft import PeftModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers.trainer_utils import get_last_checkpoint
import yaml

from src.config import (
    DPO_DATA_DIR,
    CONFIG_DIR,
    MODEL_DIR,
    MODEL_NAME,
    SFT_DIR,
    DPO_DIR,
    MAX_PROMPT_TOKENS,
)

def load_dpo_cfg():
    with open(CONFIG_DIR / "dpo.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_dpo(base_model_path, sft_dir, dpo_dir):
    cfg = load_dpo_cfg()
    tcfg = cfg.get("training", {}) if cfg else {}
    lcfg = cfg.get("lora", {}) if cfg else {}
    ctcfg = cfg.get("chat_template", {}) if cfg else {}
    max_seq_len = int(tcfg.get("max_length", MAX_PROMPT_TOKENS))

    print("加载分词器和基座模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_model_path),
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )
    
    print("设置 Qwen2.5 聊天模板...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=ctcfg.get("name", "qwen2.5"),
    )
    
    # 在 SFT 适配器基础上继续训练
    if sft_dir.exists():
        print(f"加载并继续微调 SFT 适配器: {sft_dir}")
        model = PeftModel.from_pretrained(
            model,
            str(sft_dir),
            device_map={"": torch.cuda.current_device()},
            is_trainable=True,
        )
    else:
        print("未找到 SFT 适配器，直接初始化 DPO LoRA 适配器")
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lcfg.get("r", 8)),
            target_modules=lcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_alpha=int(lcfg.get("lora_alpha", 16)),
            lora_dropout=float(lcfg.get("lora_dropout", 0)),
            bias=str(lcfg.get("bias", "none")),
            use_gradient_checkpointing="unsloth",
            random_state=42,
            max_seq_length=max_seq_len,
        )
    
    print("加载数据集...")
    train_data = load_dataset("json", data_files=str(DPO_DATA_DIR / "train_data.jsonl"), split="train")
    valid_data = load_dataset("json", data_files=str(DPO_DATA_DIR / "valid_data.jsonl"), split="train")
    print(f"训练数据: {len(train_data)} 条")
    print(f"验证数据: {len(valid_data)} 条")
    
    print("配置 DPO 训练参数...")
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        ref_model=None,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=DPOConfig(
            output_dir=str(dpo_dir),
            per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
            learning_rate=float(tcfg.get("learning_rate", 5e-6)),
            lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
            warmup_ratio=float(tcfg.get("warmup_ratio", 0.1)),
            weight_decay=float(tcfg.get("weight_decay", 0.01)),
            num_train_epochs=int(tcfg.get("num_train_epochs", 2)),
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
            metric_for_best_model=str(tcfg.get("metric_for_best_model", "eval_rewards/chosen")),
            dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 8)),
            dataloader_pin_memory=bool(tcfg.get("dataloader_pin_memory", True)),
            gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": bool(tcfg.get("gradient_checkpointing_use_reentrant", False))},
            max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
            max_length=int(tcfg.get("max_length", 2048)),
            max_prompt_length=int(tcfg.get("max_prompt_length", 1024)),
            beta=float(tcfg.get("beta", 0.1)),
            report_to=["tensorboard"],
        ),
    )
    
    print(f"开始 DPO 训练...")
    # 检查是否有 checkpoint
    last_checkpoint = get_last_checkpoint(str(dpo_dir))
    if last_checkpoint is not None:
        print(f"发现已有 DPO checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    
    print(f"保存 DPO 适配器（SFT + DPO 的组合效果）...")
    trainer.save_model(dpo_dir)
    
    # 清理资源
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    base_model_path = MODEL_DIR / MODEL_NAME 
    DPO_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*20} 开始训练 DPO 适配器 {'='*20}")
    try:
        train_dpo(base_model_path, SFT_DIR, DPO_DIR)
    except Exception as e:
        print(f"DPO 适配器训练失败: {e}")

if __name__ == "__main__":
    main()

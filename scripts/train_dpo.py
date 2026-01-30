import sys
import torch
import gc
from pathlib import Path
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DPO_DATA_DIR,
    MODEL_NAME,
    BASE_MODEL_DIR,
    SFT_ADAPTER_DIR,
    DPO_ADAPTER_DIR,
    VERSIONS,
)
from src.utils import ModelRunning

def train_dpo(base_model_path, ratio, sft_adapter_path, dpo_adapter_path):
    print("加载基座模型和分词器...")
    model_running = ModelRunning(
        base_model_path=str(base_model_path), 
        sft_adapter_path=str(sft_adapter_path)
    )
    model = model_running.model
    tokenizer = model_running.tokenizer
    
    print("将 SFT 适配器合并到基座模型...")
    model = model.merge_and_unload()
    
    # 确保 pad_token 设置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    
    training_args = DPOConfig(
        output_dir=str(dpo_adapter_path),
        per_device_train_batch_size=1,      # 每个设备的 batch size
        gradient_accumulation_steps=4,      # 梯度累积步数
        learning_rate=5e-6,                 # 学习率 (通常比 SFT 低)
        num_train_epochs=1,                 # DPO 通常只需要很少的 epoch (1-3)
        beta=0.1,                           # DPO 特有: 控制与参考模型的偏离程度 (KL 散度惩罚系数)
        logging_steps=10,                   # 打印日志频率
        save_steps=50,                      # 保存频率
        bf16=True,                          # 使用 bfloat16
        remove_unused_columns=False,        # DPO 需要保留 prompt/chosen/rejected 列
        
        # DPO 特有长度控制
        max_length=512,                    # 总最大长度 (Prompt + Response)
        max_prompt_length=256,              # Prompt 的最大长度
        
        gradient_checkpointing=True,        # 启用梯度检查点，节省显存
    )
    
    print(f"加载数据集: {raw_train}")
    raw_train = load_dataset("json", data_files=str(DPO_DATA_DIR / "dpo_data.jsonl"), split="train")
    print(f"原始训练数据: {len(raw_train)} 条")
    
    sampled_train = raw_train.shuffle(seed=42).select(range(int(len(raw_train) * ratio)))
    print(f"采样后训练数据: {len(sampled_train)} 条")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=sampled_train,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    print("开始 DPO 训练...")
    trainer.train()
    
    print(f"保存 DPO 适配器至: {dpo_adapter_path}")
    model.save_pretrained(dpo_adapter_path) 
    tokenizer.save_pretrained(dpo_adapter_path)
    
    # 清理训练资源
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    gc.collect()

def main():
    if not DPO_DATA_DIR.exists():
        print(f"错误: DPO 数据目录不存在: {DPO_DATA_DIR}")
        return
    if not MODEL_NAME:
        print(f"错误: 模型名称未配置")
        return
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    if not base_model_path.exists():
        print(f"错误: 基座模型不存在: {base_model_path}")
        return    
    DPO_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    if not VERSIONS:
        print(f"错误: 版本号未配置")
        return
    
    for version, ratio in VERSIONS:
        dpo_adapter_path = DPO_ADAPTER_DIR / version
        dpo_adapter_exists = dpo_adapter_path.exists() and (dpo_adapter_path / "adapter_config.json").exists()
        
        if dpo_adapter_exists:
            print(f"({version} 版本) DPO 适配器已存在，跳过训练")
            continue
        
        sft_adapter_path = SFT_ADAPTER_DIR / version
        if not sft_adapter_path.exists():
            print(f"({version} 版本) SFT 适配器不存在，跳过")
            continue
        print(f"\n{'='*20} 开始训练 ({version} 版本) DPO 适配器 (数据比例: {ratio}) {'='*20}")
        try:
            train_dpo(base_model_path, ratio, sft_adapter_path, dpo_adapter_path)
        except Exception as e:
            print(f"({version} 版本) DPO 适配器训练失败: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n训练结束！")

if __name__ == "__main__":
    main()

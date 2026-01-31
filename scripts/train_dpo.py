import sys
import torch
import gc
from pathlib import Path
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DPO_DATA_DIR,
    MODEL_NAME,
    BASE_MODEL_DIR,
    SFT_ADAPTER_DIR,
    DPO_ADAPTER_DIR,
    VERSIONS,
    QUANT_CONFIG,
    LORA_CONFIG,
)

_tokenizer = None

def get_tokenizer(base_model_path: Path) -> AutoTokenizer:
    """获取或创建tokenizer（单例模式）"""
    global _tokenizer
    if _tokenizer is None:
        print("初始化分词器...")
        _tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "right"
        _tokenizer.truncation_side = "right"
    return _tokenizer

def train_dpo(base_model_path, ratio, sft_adapter_path, dpo_adapter_path):
    print("加载分词器...")
    _tokenizer = get_tokenizer(base_model_path)
    
    print("加载模型...")
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            quantization_config=QUANT_CONFIG,
            device_map={"": "cuda"},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ),
        str(sft_adapter_path),
        device_map={"": "cuda"},
    )
    
    print("将 SFT 适配器合并到基座模型...")
    model = model.merge_and_unload()
    
    dpo_args = DPOConfig(
        output_dir=str(dpo_adapter_path),
        per_device_train_batch_size=1,      # 每个设备的训练 batch size
        per_device_eval_batch_size=1,       # 每个设备的验证 batch size
        gradient_accumulation_steps=8,      # 梯度累积步数，模拟更大的 batch size
        learning_rate=5e-5,                 # 学习率
        num_train_epochs=2,                 # 训练轮数
        logging_steps=10,                   # 打印日志频率
        save_steps=100,                     # 保存模型检查点频率
        eval_steps=100,                     # 评估模型频率
        evaluation_strategy="steps",        # 评估策略
        load_best_model_at_end=True,        # 训练结束后加载最优模型
        metric_for_best_model="eval_reward/chosen",  # 用于选择最优模型的指标
        fp16=torch.cuda.is_available(),     # 使用 float16 精度
        warmup_ratio=0.1,                   # 预热比例
        weight_decay=0.01,                  # 权重衰减
        report_to=[],                       # 不上报到 wandb 等平台
        gradient_checkpointing=True,        # 启用梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},# 设置非重入模式
        optim="adamw_8bit",                 # 使用8位优化器
        group_by_length=False,              # 按长度分组
    )
    
    print(f"加载数据集...")
    raw_train = load_dataset("json", data_files=str(DPO_DATA_DIR / "train_data.jsonl"), split="train")
    raw_valid = load_dataset("json", data_files=str(DPO_DATA_DIR / "valid_data.jsonl"), split="train")
    print(f"原始训练数据: {len(raw_train)} 条")
    print(f"原始验证数据: {len(raw_valid)} 条")
    
    sampled_train = raw_train.shuffle(seed=42).select(range(int(len(raw_train) * ratio)))
    sampled_valid = raw_valid.shuffle(seed=42).select(range(int(len(raw_valid) * ratio)))
    print(f"采样训练数据: {len(sampled_train)} 条")
    print(f"采样验证数据: {len(sampled_valid)} 条")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=sampled_train,
        eval_dataset=sampled_valid,
        peft_config=LORA_CONFIG,
        max_length=512,
        max_prompt_length=256,
        tokenizer=_tokenizer,
        args=dpo_args,
    )

    print(f"开始 DPO 训练...")
    trainer.train()
    
    print(f"保存 DPO 适配器...")
    trainer.save_model(dpo_adapter_path)
    
    # 清理资源
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    if not DPO_DATA_DIR.exists():
        print(f"错误: DPO 数据目录不存在: {DPO_DATA_DIR}")
        return
    if not MODEL_NAME:
        print(f"错误: 模型名称未配置")
        return
    if not BASE_MODEL_DIR:
        print("错误: 基座模型目录未配置")
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

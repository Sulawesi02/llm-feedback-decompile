from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import os

# 数据路径
data_files = {
    "train": "/app/dataset/exebench/train/train_machine_to_c.jsonl",
    "valid": "/app/dataset/exebench/valid/valid_machine_to_c.jsonl"
}
dataset = load_dataset("json", data_files=data_files)

# 模型名称
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 输出路径
LORA_PATH = "/app/model/final/machine_to_c_lora"      # LoRA 适配器
MERGED_PATH = "/app/model/final/machine_to_c_merged"  # 合并后的完整模型

os.makedirs(LORA_PATH, exist_ok=True)
os.makedirs(MERGED_PATH, exist_ok=True)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Prompt 模板
def formatting_prompts_func(examples):
    texts = []
    for arch, opt, machine_code, c_code in zip(
        examples['arch'], examples['opt'], examples['machine_code'], examples['c_code']
    ):
        prompt = f"""你是一个专业的二进制反编译专家。
请把下面这段 {arch} {opt} 机器码反编译成可读的 C 语言函数：

{machine_code}

输出只包含 C 代码，不要解释，不要添加额外文字。

C 代码：
{c_code}"""
        texts.append(prompt)
    return {"text": texts}

# 训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=3e-4,
    num_train_epochs=3,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    logging_steps=20,
    output_dir=LORA_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    seed=3407,
    report_to=[],
    disable_tqdm=False,
    dataloader_num_workers=0,
    torch_compile=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
)

# 开始训练
trainer.train()

# 保存 LoRA 适配器（可选备份）
print("保存 LoRA 适配器")
model.save_pretrained(LORA_PATH)
tokenizer.save_pretrained(LORA_PATH)

# 合并权重
print("合并 LoRA 到基座模型")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载训练好的 LoRA 适配器
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 合并并卸载 LoRA
merged_model = model.merge_and_unload(progressbar=True)

# 保存合并后的完整模型
print("保存合并后的完整模型")
merged_model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print(f"训练完成！")
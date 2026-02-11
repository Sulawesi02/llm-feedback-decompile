import json
import sys
import torch
import gc
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    PROCESSED_DATA_DIR, 
    EVAL_DIR,
    MODEL_NAME,
    BASE_MODEL_DIR, 
    SFT_ADAPTER_DIR,
    DPO_ADAPTER_DIR,
    VERSIONS,
    QUANT_CONFIG,
    MAX_CONTEXT_TOKENS,
    MAX_PROMPT_TOKENS,
)
from src.compiler import (
    test_func,
)
from src.prompts import (
    construct_infer_prompt,
    construct_fix_prompt,
)

MAX_SAMPLES = 100 # 最大样本数
MAX_ITERS = 3 # 最大迭代次数

_tokenizer = None
_basemodel = None

def get_tokenizer(base_model_path: Path):
    """获取或创建tokenizer（单例模式）"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "right"
        _tokenizer.truncation_side = "right"
    return _tokenizer

def get_basemodel(base_model_path: Path):
    """获取或创建基座模型（单例模式）"""
    global _basemodel
    if _basemodel is None:
        _basemodel = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            quantization_config=QUANT_CONFIG,
            device_map={"": torch.cuda.current_device()},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        _basemodel.config.use_cache = False
    return _basemodel
    
def process_single_sample(model, func_dep: str, func: str, test: str, asm: str):
    """
    处理单条样本
    """
    # print(f"输入 C 函数:\n{func}")
    gen_outputs = None
    prev_outputs = None
    last_error = None
    
    history = []
    best_outputs = None
    
    for it in range(MAX_ITERS):
        print(f"\n--- 迭代 {it+1}/{MAX_ITERS} ---")
        
        try:
            # ========== 生成阶段 ==========
            if it == 0:
                print("构造推理提示...")
                gen_messages = construct_infer_prompt(asm)
            else:
                print("构造修复提示...")
                gen_messages = construct_fix_prompt(asm, prev_outputs, last_error)
            gen_text = _tokenizer.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)            
            
            gen_inputs_check = _tokenizer(
                gen_text, 
                return_tensors="pt", 
                truncation=False
            )
            gen_prompt_tokens = gen_inputs_check.input_ids.shape[1]
            if gen_prompt_tokens > MAX_PROMPT_TOKENS:
                print(f"输入长度 {gen_prompt_tokens} 超过最大长度")
                error_msg = f"生成提示过长: {gen_prompt_tokens} tokens"
                history.append({
                    "iter": it,
                    "success": False,
                    "outputs": "",
                    "error": error_msg,
                })
                break
            
            gen_inputs = _tokenizer(
                gen_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=gen_prompt_tokens
            ).to(model.device)

            print(f"生成 C 函数代码...")
            gen_avail_tokens = MAX_CONTEXT_TOKENS - gen_prompt_tokens
            with torch.no_grad():
                gen_outputs = model.generate(
                    **gen_inputs,
                    max_new_tokens=gen_avail_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=_tokenizer.eos_token_id,
                    pad_token_id=_tokenizer.pad_token_id,
                )
            gen_output_text = _tokenizer.decode(gen_outputs[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("```cpp", "").replace("```c", "").replace("```", "").strip()
            print(f"生成的 C 函数代码:\n{gen_output_text}")
            # ========== 测试阶段 ==========
            print("测试 C 函数代码...")
            success, error_msg = test_func(func_dep, gen_output_text, test)
            if success:
                print(f"  测试通过！")
                best_outputs = gen_output_text
                history.append({
                    "iter": it,
                    "success": True,
                    "outputs": gen_output_text,
                    "error": "",
                })
                break
            else:
                print(f"  测试失败: {error_msg.splitlines()[0]}")
                history.append({
                    "iter": it,
                    "success": False,
                    "outputs": gen_output_text,
                    "error": error_msg,
                })
                prev_outputs = gen_output_text
                last_error = error_msg
                continue
        except Exception as e:
            print(f"生成 C 函数代码出错: {e}")
            history.append({
                "iter": it,
                "success": False,
                "outputs": "",
                "error": f"生成 C 函数代码出错: {e}",
            })
            break
    print(f"任务结束 | 最终状态: {'成功' if best_outputs else '失败'}")
    result_entry = {
        "func": func,
        "asm": asm,
        "success": best_outputs is not None,
        "best_outputs": best_outputs,
        "history": history,
    }
    return result_entry

def evaluate_model(test_input: Path, base_model_path: Path, dpo_adapter_path: Optional[Path], eval_out_path: Path):
    
    global _tokenizer, _basemodel
    
    success_count = 0
    total_count = 0
    processed_keys = set()
    existing_results = []

    # 读取已有记录以支持断点续传
    if eval_out_path.exists():
        try:
            with open(eval_out_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    existing_results = []
                else:
                    for line in content.splitlines():
                        if line.strip():
                            existing_results.append(json.loads(line))
            
            print(f"发现 {len(existing_results)} 条已有记录")
            
            # 统计已有记录
            for item in existing_results:
                key = item.get("func")
                if key:
                    processed_keys.add(key)
                if item.get("success"):
                    success_count += 1
                total_count += 1
                
        except Exception as e:
            print(f"读取现有结果失败: {e}, 将重新开始")
            existing_results = []

    # 检查是否已达到目标记录数
    target_count = MAX_SAMPLES * 2
    if total_count >= target_count:
        print(f"已达到目标记录数 ({MAX_SAMPLES} 条样本 -> {target_count} 条记录)，跳过评估")
        accuracy = success_count / total_count if total_count > 0 else 0
        print(f"总记录数: {total_count}")
        print(f"成功编译数: {success_count}")
        print(f"编译通过率: {accuracy:.2%}")
        return

    print(f"当前进度: {total_count}/{target_count}, 继续评估...")

    print("加载分词器...")
    _tokenizer = get_tokenizer(base_model_path)
    
    print("加载模型...")
    _basemodel = get_basemodel(base_model_path)
    model = _basemodel
    if dpo_adapter_path:
        print(f"加载 DPO 适配器: {dpo_adapter_path}")
        model = PeftModel.from_pretrained(
            _basemodel,
            str(dpo_adapter_path),
            device_map={"": torch.cuda.current_device()},
        )
    model.eval() # 推理模式

    if existing_results:
        with open(eval_out_path, "w", encoding="utf-8") as f:
            for item in existing_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f_out = open(eval_out_path, "a", encoding="utf-8")
    else:
        # 新文件
        f_out = open(eval_out_path, "w", encoding="utf-8")

    test_data = load_dataset("json", data_files=str(test_input), split="train")
    sampled_test = test_data.select(range(MAX_SAMPLES))

    try:
        for item in tqdm(sampled_test, desc="评估进度"):
            func_dep = item.get("func_dep")
            func = item.get("func")
            test = item.get("test")
            asm = item.get("asm")
            
            if not func or not asm or not test:
                print("Skipping invalid item")
                continue

            # 检查是否已处理
            key = func
            if key in processed_keys:
                continue

            result_entry = process_single_sample(model, func_dep, func, test, asm)
            
            f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            f_out.flush()
            
            if result_entry.get("success"):
                success_count += 1
            total_count += 1
    finally:
        f_out.close()

    accuracy = success_count / total_count if total_count > 0 else 0
    print(f"总样本数: {total_count}")
    print(f"成功编译数: {success_count}")
    print(f"编译通过率: {accuracy:.2%}")

    # 清理资源
    del model
    gc.collect()
    torch.cuda.empty_cache()

def main():
    test_input = PROCESSED_DATA_DIR / "test_data.jsonl"
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 评估基座模型
    base_eval_out = EVAL_DIR / "base_model.jsonl"
    print(f"{'='*20} 开始评估基座模型 {'='*20}")
    try:
        evaluate_model(test_input, base_model_path, None, base_eval_out)
    except Exception as e:
        print(f"基座模型评估失败: {e}")

    # 2. 评估各个版本
    for version, _ in VERSIONS:
        eval_out_path = EVAL_DIR / f"{version}.jsonl"
        sft_adapter_path = SFT_ADAPTER_DIR / version
        if not sft_adapter_path.exists():
            print(f"错误: SFT 适配器不存在: {sft_adapter_path}")
            continue
        dpo_adapter_path = DPO_ADAPTER_DIR / version
        if not dpo_adapter_path.exists():
            print(f"错误: DPO 适配器不存在: {dpo_adapter_path}")
            continue
        print(f"{'='*20} 开始评估 ({version} 版本) {'='*20}")
        try:
            evaluate_model(test_data, base_model_path, dpo_adapter_path, eval_out_path)
        except Exception as e:
            print(f"{version} 版本模型评估失败: {e}")

if __name__ == "__main__":
    main()

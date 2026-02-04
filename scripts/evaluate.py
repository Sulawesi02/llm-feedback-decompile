import json
import sys
import re
import shutil
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
    TEST_DATA, 
    EVAL_DIR,
    MODEL_NAME,
    BASE_MODEL_DIR, 
    SFT_ADAPTER_DIR,
    DPO_ADAPTER_DIR,
    VERSIONS,
    QUANT_CONFIG,
)
from src.utils import (
    extract_data, 
    compile_to_obj,
)
from src.prompts import (
    construct_infer_prompt,
    construct_fix_prompt,
    construct_equal_prompt
)

MAX_SAMPLES = 200 # 最大样本数
MAX_ITERS = 3 # 最大迭代次数

_tokenizer = None
_model = None

def get_tokenizer(base_model_path: Path):
    global _tokenizer
    if _tokenizer is None:
        print("加载分词器...")
        _tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "right"
        _tokenizer.truncation_side = "right"
    return _tokenizer

def process_single_sample(c_code: str, arch: str, asm: str, machine_code: str):
    """
    处理单条样本
    """
    print(f"输入 C 代码:\n{c_code}")
    gen_outputs = None
    previous_outputs = None
    last_error = None
    
    history = []
    best_outputs = None
    
    for it in range(MAX_ITERS):
        print(f"\n--- 迭代 {it+1}/{MAX_ITERS} ---")
        
        o_path = None
        try:
            if it == 0:
                print("构造推理提示...")
                messages = construct_infer_prompt(arch, asm)
            else:
                print("构造修复提示...")
                messages = construct_fix_prompt(arch, asm, previous_outputs, last_error)
            text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            gen_inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(_model.device)

            print("生成 C 函数代码...")
            with torch.no_grad():
                gen_outputs = _model.generate(
                    **gen_inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=_tokenizer.eos_token_id,
                    pad_token_id=_tokenizer.pad_token_id,
                )
            gen_outputs = _tokenizer.decode(gen_outputs[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("```c", "").replace("```", "").strip()
            print(f"生成的 C 函数代码:\n{gen_outputs}")

            print("编译 C 函数代码...")
            success, error_msg, o_path = compile_to_obj(arch, gen_outputs)
            if success:
                print("编译成功，进行语义等价性判定...")
                
                messages = construct_equal_prompt(c_code, gen_outputs)
                text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                eq_inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(_model.device)
                
                with torch.no_grad():
                    eq_outputs = _model.generate(
                        **eq_inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True,
                        eos_token_id=_tokenizer.eos_token_id,
                        pad_token_id=_tokenizer.pad_token_id,
                    )
                eq_outputs = _tokenizer.decode(eq_outputs[0][eq_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("```c", "").replace("```", "").strip()
                print(f"语义等价性判定结果: {eq_outputs}")
                
                if eq_outputs == "0":
                    print("判定通过：语义等价")
                    best_outputs = gen_outputs
                    history.append({
                        "iter": it,
                        "success": True,
                        "outputs": gen_outputs,
                        "error": "",
                    })
                    break
                else:
                    print("判定失败：语义不等价")
                    error_msg = "判定失败：语义不等价"
                    history.append({
                        "iter": it,
                        "success": False,
                        "outputs": gen_outputs,
                        "error": error_msg,
                    })
                    previous_outputs = gen_outputs
                    last_error = error_msg
                    continue
            else:   
                error_msg = error_msg[:1000]
                print(f"编译失败: {error_msg.splitlines()[0] if error_msg else 'Unknown'}...")
                history.append(
                    {
                        "iter": it,
                        "success": False,
                        "outputs": gen_outputs,
                        "error": error_msg,
                    }
                )
                previous_outputs = gen_outputs
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
        finally:
            if o_path:
                workdir = Path(o_path).parent
                if workdir and workdir.exists():
                    shutil.rmtree(workdir, ignore_errors=True)

    print(f"任务结束 | 最终状态: {'成功' if best_outputs else '失败'}")
    result_entry = {
        "c_code": c_code,
        "arch": arch,
        "asm": asm,
        "machine_code": machine_code,
        "success": best_outputs is not None,
        "best_outputs": best_outputs,
        "history": history,
    }
    return result_entry

def evaluate_model(base_model_path: Path, dpo_adapter_path: Optional[Path], eval_out_path: Path):
    
    global _tokenizer, _model
    
    print("加载分词器...")
    _tokenizer = get_tokenizer(base_model_path)
    
    print("加载模型...")
    _model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        quantization_config=QUANT_CONFIG,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if dpo_adapter_path:
        print(f"加载 DPO 适配器: {dpo_adapter_path}")
        _model = PeftModel.from_pretrained(
            _model,
            str(dpo_adapter_path),
            device_map={"": torch.cuda.current_device()},
        )
    _model.eval() # 推理模式

    results = []
    success_count = 0
    total_count = 0

    raw_test = load_dataset("json", data_files=str(TEST_DATA), split="train")
    sampled_test = raw_test.select(range(MAX_SAMPLES))

    for item in tqdm(sampled_test, desc="评估进度"):
        for c_code, arch, asm, machine_code in extract_data(item):
            result_entry = process_single_sample(c_code, arch, asm, machine_code)
            results.append(result_entry)
            if result_entry.get("success"):
                success_count += 1
            total_count += 1

    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = success_count / total_count if total_count > 0 else 0
    print(f"总样本数: {total_count}")
    print(f"成功编译数: {success_count}")
    print(f"编译通过率: {accuracy:.2%}")

    # 清理资源
    del _model
    gc.collect()
    torch.cuda.empty_cache()

def main():
    if not TEST_DATA.exists():
        print(f"错误: 测试集不存在: {TEST_DATA}")
        return
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    if not base_model_path.exists():
        print(f"错误: 基座模型不存在: {base_model_path}")
        return
    if not SFT_ADAPTER_DIR.exists():
        print(f"错误: SFT 适配器目录不存在: {SFT_ADAPTER_DIR}")
        return
    if not DPO_ADAPTER_DIR.exists():
        print(f"错误: DPO 适配器目录不存在: {DPO_ADAPTER_DIR}")
        return
    if not VERSIONS:
        print(f"错误: 版本号未配置")
        return
    
    # 1. 评估基座模型
    base_eval_out = EVAL_DIR / "base_model.jsonl"
    if base_eval_out.exists():
        print("基座模型评估结果已存在，跳过")
    else:
        print(f"\n{'='*20} 开始评估基座模型 {'='*20}")
        try:
            evaluate_model(base_model_path, None, base_eval_out)
        except Exception as e:
            print(f"基座模型评估失败: {e}")

    # 2. 评估各个版本
    for version, _ in VERSIONS:
        eval_out_path = EVAL_DIR / f"{version}.jsonl"
        if eval_out_path.exists():
            print(f"({version} 版本) 评估结果已存在，跳过")
            continue
        
        sft_adapter_path = SFT_ADAPTER_DIR / version
        if not sft_adapter_path.exists():
            print(f"错误: SFT 适配器不存在: {sft_adapter_path}")
            continue
        dpo_adapter_path = DPO_ADAPTER_DIR / version
        if not dpo_adapter_path.exists():
            print(f"错误: DPO 适配器不存在: {dpo_adapter_path}")
            continue
        print(f"\n{'='*20} 开始评估 ({version} 版本) {'='*20}")
        try:
            evaluate_model(base_model_path, dpo_adapter_path, eval_out_path)
        except Exception as e:
            print(f"{version} 版本模型评估失败: {e}")

if __name__ == "__main__":
    main()

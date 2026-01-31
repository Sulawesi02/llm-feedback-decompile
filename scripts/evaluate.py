import json
import sys
import re
import shutil
import gc
import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
    machine_code_to_binary,
    disasm_binary,
    extract_asm,
    compile_to_object,
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

def evaluate_single_task(sample_index: int, task: dict):
    """
    评估单个任务
    """
    arch = task["arch"]
    machine_code = task["machine_code"]
    c_code = task["c_code"]
    
    print(f"\n{'='*50}")
    print(f"开始评估: 样本 {sample_index} | 架构: {arch}")
    print(f"{'='*50}")

    outputs = None
    previous_outputs = None
    last_error = None
    
    history = []
    best_outputs = None
    
    for it in range(MAX_ITERS):
        print(f"\n--- 迭代 {it+1}/{MAX_ITERS} ---")
        
        o_path = None
        try:
            binary_path = machine_code_to_binary(machine_code)            
            disasm_result = disasm_binary(arch, binary_path)
            asm = extract_asm(arch, disasm_result)

            if it == 0:
                print("构造推理提示...")
                messages = construct_infer_prompt(arch, asm)
            else:
                print("构造修复提示...")
                messages = construct_fix_prompt(arch, asm, previous_outputs, last_error)
            
            gen_inputs = _tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            print("生成 C 函数代码...")
            with torch.no_grad():
                gen_outputs = _model.generate(
                    gen_inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=_tokenizer.eos_token_id,
                    pad_token_id=_tokenizer.pad_token_id,
                )
            gen_outputs = _tokenizer.decode(gen_outputs[0], skip_special_tokens=True).split("assistant\n")[-1]

            print("编译 C 函数代码...")
            success, error_msg, o_path = compile_to_object(arch, gen_outputs)
            if success:
                print("编译成功，进行语义等价性判定...")
                
                eval_messages = construct_equal_prompt(c_code, gen_outputs)
                eq_inputs = _tokenizer.apply_chat_template(
                    eval_messages, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                
                with torch.no_grad():
                    eq_outputs = _model.generate(
                        eq_inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True,
                        eos_token_id=_tokenizer.eos_token_id,
                        pad_token_id=_tokenizer.pad_token_id,
                    )
                judgment = _tokenizer.decode(eq_outputs[0], skip_special_tokens=True).split("assistant\n")[-1]
                judgment = re.sub(r'[^0-1]', '', judgment)
                
                if judgment == "0":
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
                    break
            else:   
                error_msg = error_msg[:1000]
                print(f"编译失败: {error_msg.splitlines()[0] if error_msg else 'Unknown'}...")
                history.append(
                    {
                        "iter": it,
                        "success": False,
                        "outputs": outputs,
                        "error": error_msg,
                    }
                )
                previous_outputs = outputs
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
        "id": f"{sample_index}_{task['id_suffix']}",
        "arch": arch,
        "success": best_outputs is not None,
        "machine_code": task["machine_code"],
        "c_code": task["c_code"],
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
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if dpo_adapter_path:
        print(f"加载 DPO 适配器: {dpo_adapter_path}")
        _model = PeftModel.from_pretrained(
            _model,
            str(dpo_adapter_path),
            device_map={"": "cuda:0"},
        )
    else:
        print("未提供适配器路径，使用基座模型进行评估")

    _model.eval() # 推理模式

    results = []
    success_count = 0
    total_count = 0

    with open(TEST_DATA, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = lines[:MAX_SAMPLES]

    for i, line in tqdm(enumerate(lines), total=len(lines), desc="评估进度"):
        try:
            data = json.loads(line)
            tasks = []
            for arch, arch_data in data["compilations"].items():
                tasks.append(
                    {
                        "arch": arch,
                        "machine_code": arch_data.get("machine_code"),
                        "c_code": data.get("c_code"),
                        "id_suffix": arch,
                    }
                )

            for task in tasks:
                result_entry = evaluate_single_task(i, task)
                results.append(result_entry)
                if result_entry["success"]:
                    success_count += 1
                total_count += 1

        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue

    Path(eval_out_path).parent.mkdir(parents=True, exist_ok=True)
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
            import traceback
            traceback.print_exc()

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
            
        try:
            evaluate_model(base_model_path, dpo_adapter_path, eval_out_path)
        except Exception as e:
            print(f"{version} 版本模型评估失败: {e}")

if __name__ == "__main__":
    main()

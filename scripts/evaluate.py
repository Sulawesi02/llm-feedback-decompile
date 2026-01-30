import os
import json
import sys
import re
import time
import shutil
from pathlib import Path
from tqdm import tqdm

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
    OFFLOAD_ROOT,
)
from src.utils import (
    ModelRunning,
    machine_code_to_binary,
    disasm_binary,
    extract_asm,
    clean_code_block,
    compile_to_object,
)
from src.prompts import (
    construct_infer_prompt,
    construct_fix_prompt,
    construct_equivalence_prompt
)

MAX_SAMPLES = 200 # 最大样本数
MAX_ITERS = 3 # 最大迭代次数

model_runner = None

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

    generate_c_code = None
    previous_c_code = None
    last_error = None
    
    history = []
    best_c_code = None
    
    start_time = time.time()
    TOTAL_TIMEOUT = 120.0

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
                messages = construct_fix_prompt(arch, asm, previous_c_code, last_error)
            
            run_time = time.time() - start_time
            remaining_time = TOTAL_TIMEOUT - run_time
            print("生成 C 函数代码...")
            generate_c_code = model_runner.generate(messages, remaining_time)
            generate_c_code = clean_code_block(generate_c_code)

            print("编译 C 函数代码...")
            success, error_msg, o_path = compile_to_object(arch, generate_c_code)
            
            if success:
                print("编译成功，进行语义等价性判定...")
                
                eval_messages = construct_equivalence_prompt(c_code, generate_c_code)
                judgment = model_runner.generate(eval_messages, remaining_time=30.0)
                judgment = judgment.strip()
                judgment = re.sub(r'[^0-1]', '', judgment)
                
                if judgment == "0":
                    print("判定通过：语义等价")
                    best_c_code = generate_c_code
                    history.append({
                        "iter": it,
                        "success": True,
                        "generate_c_code": generate_c_code,
                        "error": "",
                    })
                    break
                else:
                    print("判定失败：语义不等价")
                    error_msg = "判定失败：语义不等价"
                    history.append({
                        "iter": it,
                        "success": False,
                        "generate_c_code": generate_c_code,
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
                        "generate_c_code": generate_c_code,
                        "error": error_msg,
                    }
                )
                previous_c_code = generate_c_code
                last_error = error_msg
                continue
        except Exception as e:
            print(f"生成 C 函数代码出错: {e}")
            history.append({
                "iter": it,
                "success": False,
                "generate_c_code": "",
                "error": f"生成 C 函数代码出错: {e}",
            })
            break
        finally:
            if o_path:
                workdir = Path(o_path).parent
                if workdir and workdir.exists():
                    shutil.rmtree(workdir, ignore_errors=True)

    print(f"任务结束 | 最终状态: {'成功' if best_c_code else '失败'}")
    result_entry = {
        "id": f"{sample_index}_{task['id_suffix']}",
        "arch": arch,
        "success": best_c_code is not None,
        "machine_code": task["machine_code"],
        "c_code": task["c_code"],
        "best_c_code": best_c_code,
        "history": history,
    }
    return result_entry

def evaluate_model(base_model_path: Path, sft_adapter_path: Path, dpo_adapter_path: Path, version: str, eval_out_dir: Path):
    eval_out_path = eval_out_dir / f"{version}.jsonl"
    
    global model_runner
    
    offload_dir = OFFLOAD_ROOT / "evaluate"
    offload_dir.mkdir(parents=True, exist_ok=True)

    model_runner = ModelRunning(
        base_model_path=str(base_model_path),
        sft_adapter_path=str(sft_adapter_path),
        dpo_adapter_path=str(dpo_adapter_path),
        offload_folder=str(offload_dir),
        offload_buffers=False
    )

    results = []
    success_count = 0
    total_count = 0

    try:
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

    finally:
        model_runner.unload()

    Path(eval_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = success_count / total_count if total_count > 0 else 0
    print(f"\n评估完成: {version}")
    print(f"总样本数: {total_count}")
    print(f"成功编译数: {success_count}")
    print(f"编译通过率: {accuracy:.2%}")

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
    
    eval_out_dir = EVAL_DIR / MODEL_NAME
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    
    for version, _ in VERSIONS:
        sft_adapter_path = SFT_ADAPTER_DIR / version
        if not sft_adapter_path.exists():
            print(f"错误: SFT 适配器不存在: {sft_adapter_path}")
            continue
        dpo_adapter_path = DPO_ADAPTER_DIR / version
        if not dpo_adapter_path.exists():
            print(f"错误: DPO 适配器不存在: {dpo_adapter_path}")
            continue
            
        try:
            evaluate_model(base_model_path, sft_adapter_path, dpo_adapter_path, version, eval_out_dir)
        except Exception as e:
            print(f"{version} 版本模型评估失败: {e}")

if __name__ == "__main__":
    main()

import os
import json
import shutil
import sys
import re
import time
from pathlib import Path
from typing import List
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BASE_MODEL_DIR_PATH, MERGED_MODEL_DIR_PATH, TEST_DATA_FILE_PATH, EVAL_DIR
from src.utils import (
    ModelRunner,
    machine_code_to_binary,
    disasm_binary,
    extract_asm,
    construct_decompile_prompt,
    clean_code_block,
    compile_to_object,
    build_test_harness,
    run_test_harness,
    extract_function_signature,
)

# MODEL_PATH = str(BASE_MODEL_DIR_PATH / "Qwen2.5-Coder-7B-Instruct")
MODEL_PATH = str(MERGED_MODEL_DIR_PATH / "Qwen2.5-Coder-7B-Instruct" / "v1")
MAX_SAMPLES = 200 # 最大样本数
MAX_ITERS = 3 # 最大迭代次数

model_runner = None

def is_simple_literal(val: str) -> bool:
    """
    检查是否为简单字面量
    """
    # 1. 布尔值
    if val.lower() in ("true", "false"):
        return True
    
    # 2. 空指针常量
    if val in ("NULL", "nullptr", "0", "null"):
        return True
    
    # 3. 整数
    # 十进制整数
    if re.fullmatch(r"-?\d+", val):
        return True
    # 二进制整数（0b或0B开头）
    if re.fullmatch(r"-?0[bB][01]+", val):
        return True
    # 八进制整数（0开头）
    if re.fullmatch(r"-?0[0-7]+", val):
        return True    
    # 十六进制整数（0x或0X开头）
    if re.fullmatch(r"-?0[xX][0-9a-fA-F]+", val):
        return True
    
    # 4. 浮点数
    # 标准形式
    if re.fullmatch(r"-?\d+\.\d*", val):
        return True
    # 省略整数部分
    if re.fullmatch(r"-?\.\d+", val):
        return True
    # 科学计数法
    if re.fullmatch(r"-?\d+\.\d*[eE][+-]?\d+", val):
        return True
    if re.fullmatch(r"-?\.\d+[eE][+-]?\d+", val):
        return True
    # 整数科学计数法
    if re.fullmatch(r"-?\d+[eE][+-]?\d+", val):
        return True
    
    # 5. 字符
    if re.match(r"^'.'$", val):
        return True
    
    # 6. 字符串
    if re.match(r'^".*"$', val):
        return True

    return False

def has_meaningful_return(code: str) -> bool:
    """
    检查代码是否有有意义的返回值。
    如果返回类型是 void，或者只返回常数/空，则认为无意义。
    """
    sig = extract_function_signature(code)
    if sig:
        rt = sig["return_type"]
        # void 且非指针
        if re.search(r'\bvoid\b', rt) and "*" not in rt:
            return False

    for raw_line in code.splitlines():
        line = raw_line.strip()
        
        # 忽略注释
        if line.startswith("//") or line.startswith("/*") or line.startswith("*"):
            continue
        
        # 筛选 return 语句
        if not re.match(r"^return\b", line):
            continue
        
        # 提取返回值
        match = re.match(r"^return\b\s*(.*?)\s*;?\s*$", line)
        if not match:
            continue
        
        # 移除可能的空格
        val = match.group(1).strip()
        if not val:
            continue
        
        # 去除可能的括号 return (0);
        while val.startswith("(") and val.endswith(")"):
            val = val[1:-1].strip()
        
        # 简单字面量检查
        if is_simple_literal(val):
            continue
        return True
    return False

def parse_generated_tests(text: str) -> List[dict]:
    """
    从模型生成的测试用例文本中解析出测试用例
    """
    cases: List[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        if "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        input_str = left.strip()
        output_str = right.strip()
        if not input_str or not output_str:
            continue
        args = [a.strip() for a in input_str.split(",") if a.strip()]
        if not args:
            continue
        cases.append({"args": args, "expected": output_str})
    return cases

def evaluate_single_task(sample_index: int, task: dict):
    """
    评估单个任务
    """
    arch = task["arch"]
    machine_code = task["machine_code"]
    source_c_code = task["source_c_code"]
    
    print(f"\n{'='*50}")
    print(f"开始评估: 样本 {sample_index} | 架构: {arch}")
    print(f"{'='*50}")

    c_code = None
    previous_c_code = None
    last_error = None
    
    history = []
    best_c_code = None
    
    start_time = time.time()
    TOTAL_TIMEOUT = 120.0

    for it in range(MAX_ITERS):
        print(f"\n--- 迭代 {it+1}/{MAX_ITERS} ---")
        
        elapsed = time.time() - start_time
        remaining = TOTAL_TIMEOUT - elapsed
        if remaining <= 0:
            print("总处理时间超时")
            history.append({
                "iter": it,
                "step": "timeout",
                "status": "error",
                "c_code": "",
                "message": "处理超时",
                "error": "处理超时",
                "test_stdout": "",
                "test_stderr": "",
            })
            break
        
        binary_path = machine_code_to_binary(machine_code)
        if not binary_path:
            print("机器码转换为二进制失败")
            history.append({
                "iter": it,
                "step": "binary",
                "status": "error",
                "c_code": "",
                "message": "机器码转换为二进制失败",
                "error": "机器码转换为二进制失败",
                "test_stdout": "",
                "test_stderr": "",
            })
            break
        
        disasm_result = disasm_binary(arch, binary_path)
        if not disasm_result:
            print("反汇编二进制文件失败")
            history.append({
                "iter": it,
                "step": "disasm",
                "status": "error",
                "c_code": "",
                "message": "反汇编二进制文件失败",
                "error": "反汇编二进制文件失败",
                "test_stdout": "",
                "test_stderr": "",
            })
            break
        asm = extract_asm(arch, disasm_result)
        if it == 0:
            print("生成初始 C 函数代码...")
            messages = construct_decompile_prompt(arch, asm)
        else:
            print("生成修复 C 函数代码...")
            messages = construct_decompile_prompt(arch, asm, previous_c_code, last_error)
        c_code = model_runner.generate(messages, timeout=remaining)
        c_code = clean_code_block(c_code)
        print("C 函数代码生成完成")
        history.append({
            "iter": it,
            "step": "generate" if it == 0 else "refine",
            "status": "generated",
            "c_code": c_code,
            "message": "生成初始 C 函数代码" if it == 0 else "生成修复 C 函数代码",
            "error": "",
            "test_stdout": "",
            "test_stderr": "",
        })
        
        if not c_code:
            print("生成的 C 函数代码为空")
            history.append({
                "iter": it,
                "step": "generate" if it == 0 else "refine",
                "status": "error",
                "c_code": "",
                "message": "生成的 C 函数代码为空",
                "error": "生成的 C 函数代码为空",
                "test_stdout": "",
                "test_stderr": "",
            })
            break

        compile_result = compile_to_object(arch, c_code)

        try:
            if compile_result["success"]:
                print("编译成功，准备进行测试...")
                if not has_meaningful_return(source_c_code):
                    print("函数没有有意义的return，无法测试。")
                    best_c_code = c_code
                    history.append({
                        "iter": it,
                        "step": "compile",
                        "status": "compile_success",
                        "c_code": c_code,
                        "message": "编译成功",
                        "error": "",
                        "test_stdout": "",
                        "test_stderr": "",
                    })
                    break
                print("生成测试用例...")
                test_messages = [
                    {
                        "role": "system",
                        "content": "你是一个 C 语言单元测试专家，擅长为给定的函数设计高覆盖率的白盒测试用例。",
                    },
                    {
                        "role": "user",
                        "content": f"""下面是一个完整的 C 函数实现，请为这个函数生成若干个输入输出测试用例。

要求：
1. 只考虑函数的正常输入场景，不需要故意构造未定义行为。
2. 尽量覆盖不同分支和边界条件。
3. 输出格式严格为多行纯文本，每行一个用例：
   <参数1, 参数2, ...> => <期望返回值>
4. 参数列表按函数形参顺序给出，用英文逗号分隔；期望返回值用 C 表达式写法。
5. 不要输出任何解释、注释或额外文字。

函数代码：
{source_c_code}
""",
                    },
                ]
                tests_text = model_runner.generate(test_messages)
                generated_cases = parse_generated_tests(tests_text)
                if not generated_cases:
                    print("生成测试用例失败。")
                    best_c_code = c_code
                    history.append({
                        "iter": it,
                        "step": "compile",
                        "status": "compile_success",
                        "c_code": c_code,
                        "message": "编译成功，生成测试用例失败",
                        "error": "",
                        "test_stdout": "",
                        "test_stderr": "",
                    })
                    continue
                print("构建测试代码...")
                harness_code = build_test_harness(c_code, generated_cases)
                if not harness_code:
                    print("构建测试代码失败")
                    history.append({
                        "iter": it,
                        "step": "test",
                        "status": "error",
                        "c_code": c_code,
                        "message": "编译成功，构建测试代码失败",
                        "error": "",
                        "test_stdout": "",
                        "test_stderr": "",
                    })
                    continue
                print("运行测试代码...")
                test_result = run_test_harness(arch, compile_result["binary_path"], harness_code)
                if test_result["success"]:
                    print("测试通过。")
                    best_c_code = c_code
                    history.append({
                        "iter": it,
                        "step": "test",
                        "status": "test_success",
                        "c_code": c_code,
                        "message": "编译成功，测试通过。",
                        "error": "",
                        "test_stdout": test_result.get("stdout", ""),
                        "test_stderr": test_result.get("stderr", ""),
                    })
                    break
                print(f"测试未通过: {test_result['error'][:100]}...")
                error_msg = test_result["error"][:1000]
                history.append({
                    "iter": it,
                    "step": "test",
                    "status": "test_failed",
                    "c_code": c_code,
                    "message": "编译成功，测试未通过",
                    "error": error_msg,
                    "test_stdout": test_result.get("stdout", ""),
                    "test_stderr": test_result.get("stderr", ""),
                })
                previous_c_code = c_code
                last_error = error_msg
                continue
            error_msg = compile_result["error"][:1000]
            print(f"编译失败: {error_msg.splitlines()[0] if error_msg else 'Unknown'}...")
            history.append(
                {
                    "iter": it,
                    "step": "compile",
                    "status": "compile_failed",
                    "c_code": c_code,
                    "message": "",
                    "error": error_msg,
                    "test_stdout": "",
                    "test_stderr": "",
                }
            )
            previous_c_code = c_code
            last_error = error_msg
            continue
        except Exception as e:
            print(f"处理样本 {sample_index} 时出错: {e}")
            continue
        finally:
            if compile_result.get("workdir") and os.path.exists(compile_result["workdir"]):
                shutil.rmtree(compile_result["workdir"], ignore_errors=True)

    is_success = best_c_code is not None
    print(f"任务结束 | 最终状态: {'成功' if is_success else '失败'}")
    result_entry = {
        "id": f"{sample_index}_{task['id_suffix']}" if task["id_suffix"] else sample_index,
        "arch": arch,
        "success": is_success,
        "machine_code": task["machine_code"],
        "source_c_code": task["source_c_code"],
        "best_c_code": best_c_code,
        "history": history,
    }
    return result_entry, is_success

def evaluate_model(output_path: str):
    global model_runner
    model_runner = ModelRunner(MODEL_PATH)

    results = []
    success_count = 0
    total_count = 0

    try:
        with open(TEST_DATA_FILE_PATH, "r", encoding="utf-8") as f:
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
                            "source_c_code": data.get("c_code"),
                            "id_suffix": arch,
                        }
                    )

                for task in tasks:
                    result_entry, is_success = evaluate_single_task(i, task)
                    results.append(result_entry)
                    if is_success:
                        success_count += 1
                    total_count += 1

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                continue

    finally:
        model_runner.unload()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = success_count / total_count if total_count > 0 else 0
    print(f"\n评估完成: {os.path.basename(MODEL_PATH)}")
    print(f"总样本数: {total_count}")
    print(f"成功编译数: {success_count}")
    print(f"编译通过率: {accuracy:.2%}")

def main():
    output_path = EVAL_DIR / f"{Path(MODEL_PATH).parent.name}_{Path(MODEL_PATH).name}.json"

    print(f"准备评估模型: {MODEL_PATH}")

    if output_path.exists():
        print(f"评估结果已存在，跳过: {output_path}")
        return

    try:
        print(f"开始评估模型: {MODEL_PATH}")
        evaluate_model(str(output_path))
    except Exception as e:
        print(f"评估模型失败: {e}")

if __name__ == "__main__":
    main()

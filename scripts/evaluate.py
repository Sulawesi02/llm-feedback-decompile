import os
import json
import shutil
import sys
import re
from pathlib import Path
from typing import List
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BASE_MODEL_DIR_PATH, MERGED_MODEL_DIR_PATH, TEST_DATA_FILE_PATH, EVAL_DIR
from src.utils import (
    ModelRunner,
    construct_decompile_prompt,
    compile_to_object,
    build_test_harness,
    run_tests_with_harness,
    construct_refine_decompile_prompt,
)

# MODEL_PATH = str(BASE_MODEL_DIR_PATH / "Qwen2.5-Coder-7B-Instruct")
MODEL_PATH = str(MERGED_MODEL_DIR_PATH / "Qwen2.5-Coder-7B-Instruct" / "v1")
MAX_SAMPLES = 50 # 最大样本数
MAX_ITERS = 3 # 最大迭代次数

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

def has_meaningful_return(code: str) -> bool:
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line.startswith("return"):
            continue
        if re.fullmatch(r"return\s*;\s*", line):
            continue
        if not line.startswith("return "):
            continue
        body = line[len("return ") :]
        if ";" in body:
            body = body.split(";", 1)[0]
        body = body.strip()
        if not body:
            continue
        if re.fullmatch(r"-?\d+", body):
            continue
        return True
    return False

def evaluate_single_task(model_runner, sample_index: int, task: dict):
    """
    评估单个任务
    """
    arch = task["arch"]
    opt = task["opt"]
    machine_code = task["machine_code"]
    source_c_code = task["source_c_code"]
    
    print(f"\n{'='*50}")
    print(f"开始评估: 样本 {sample_index} | 架构: {arch} | 优化: {opt}")
    print(f"{'='*50}")

    messages = construct_decompile_prompt(arch, opt, machine_code)
    
    print("生成初始 C 代码...")
    c_code = model_runner.generate(messages)
    print(f"初始 C 代码生成完成")

    history = []
    best_c_code = None

    history.append({
        "iter": 0,
        "step": "generate",
        "status": "generated",
        "c_code": c_code,
        "message": "初始生成",
        "error": "",
        "test_stdout": "",
        "test_stderr": "",
    })

    if not c_code:
        print("初始代码生成为空")
        result_entry = {
            "id": f"{sample_index}_{task['id_suffix']}" if task["id_suffix"] else sample_index,
            "arch": arch,
            "opt": opt,
            "success": False,
            "final_c_code": "",
            "history": history,
        }
        return result_entry, False

    for it in range(MAX_ITERS):
        print(f"\n--- 迭代 {it+1}/{MAX_ITERS} ---")
        compile_result = compile_to_object(c_code, arch, opt)
        workdir = compile_result.get("workdir")

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
                test_result = run_tests_with_harness(compile_result["binary_path"], arch, harness_code)
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
                print("根据测试错误进行代码修复...")
                refine_messages = construct_refine_decompile_prompt(c_code, error_msg)
                c_code = model_runner.generate(refine_messages)
                if not c_code:
                    print("修复后代码为空")
                    history.append({
                        "iter": it + 1,
                        "step": "refine",
                        "status": "error",
                        "c_code": "",
                        "message": "模型生成失败",
                        "error": "",
                        "test_stdout": "",
                        "test_stderr": "",
                    })
                    break
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
            print("根据编译错误进行代码修复...")
            refine_messages = construct_refine_decompile_prompt(c_code, error_msg)
            c_code = model_runner.generate(refine_messages)
            if not c_code:
                print("修复后代码为空")
                history.append({
                    "iter": it + 1,
                    "step": "refine",
                    "status": "error",
                    "c_code": "",
                    "message": "模型生成失败",
                    "error": "",
                    "test_stdout": "",
                    "test_stderr": "",
                })
                break
        finally:
            if workdir and Path(workdir).exists():
                shutil.rmtree(workdir, ignore_errors=True)

    final_c_code = best_c_code or c_code
    final_success = best_c_code is not None
    print(f"任务结束 | 最终状态: {'成功' if final_success else '失败'}")
    result_entry = {
        "id": f"{sample_index}_{task['id_suffix']}" if task["id_suffix"] else sample_index,
        "arch": arch,
        "opt": opt,
        "success": final_c_code,
        "final_c_code": final_c_code,
        "history": history,
    }
    return result_entry, final_success

def evaluate_model(model_path: str, output_path: str):
    model_runner = ModelRunner(model_path)

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
                    for opt, opt_data in arch_data.items():
                        tasks.append(
                            {
                                "arch": arch,
                                "opt": opt,
                                "machine_code": opt_data.get("machine_code"),
                                "source_c_code": data.get("c_code"),
                                "id_suffix": f"{arch}_{opt}",
                            }
                        )

                for task in tasks:
                    result_entry, final_success = evaluate_single_task(model_runner, i, task)
                    results.append(result_entry)
                    if final_success:
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
    print(f"\n评估完成: {os.path.basename(model_path)}")
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
        evaluate_model(
            MODEL_PATH,
            str(output_path)
        )
    except Exception as e:
        print(f"评估模型失败: {e}")

if __name__ == "__main__":
    main()

import uvicorn
import shutil
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from config import BASE_MODEL_DIR_PATH, MERGED_MODEL_DIR_PATH
from utils import (
    ModelRunner,
    machine_code_to_binary,
    disasm_binary,
    extract_asm,
    construct_decompile_prompt,
    clean_code_block,
    compile_to_object,
    build_test_harness,
    run_test_harness,
)

MODEL_PATH = str(MERGED_MODEL_DIR_PATH / "Qwen2.5-Coder-7B-Instruct" / "v1")
MAX_ITERS = 3 # 最大迭代次数

model_runner = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_runner
    model_runner = ModelRunner(MODEL_PATH)
    try:
        yield
    finally:
        if model_runner:
            model_runner.unload()

app = FastAPI(lifespan=lifespan)

class DecompileRequest(BaseModel):
    arch: str # x86 / arm
    machine_code: str
    test_cases: Optional[List[dict]] = None

@app.post("/decompile")
async def decompile(request: DecompileRequest):
    if not request.machine_code.strip():
        raise HTTPException(status_code=400, detail="机器码不能为空")

    machine_code = request.machine_code # 机器码
    print(f"机器码: {machine_code}")
    arch = request.arch # 架构
    raw_cases = request.test_cases or None  # 原始测试用例
    normalized_cases: List[dict] = []  # 归一化后的测试用例
    if raw_cases and isinstance(raw_cases, list):
        for case in raw_cases:
            input_str = str(case.get("input", "")).strip()
            output_str = str(case.get("output", "")).strip()
            if not input_str or not output_str:
                continue
            args = [a.strip() for a in input_str.split(",") if a.strip()]
            if not args:
                continue
            normalized_cases.append({"args": args, "expected": output_str})

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
            break

        binary_path = machine_code_to_binary(machine_code)
        disasm_result = disasm_binary(arch, binary_path)
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
            return {"final_c_code": "// 模型服务调用失败"}

        compile_result = compile_to_object(arch, c_code)

        try:
            if compile_result["success"]:
                print("编译成功，准备进行测试...")
                if not normalized_cases:
                    print("无测试用例，无法测试。")
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
                print("构建测试代码...")
                harness_code = build_test_harness(c_code, normalized_cases)
                if not harness_code:
                    print("测试代码构建失败")
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
            print(f"处理出错: {e}")
            last_error = str(e)
            continue
        finally:
            if compile_result.get("workdir") and Path(compile_result["workdir"]).exists():
                shutil.rmtree(compile_result["workdir"], ignore_errors=True)
    
    is_success = best_c_code is not None
    print(f"任务结束 | 最终状态: {'成功' if is_success else '失败'}")
    return {
        "success": is_success,
        "best_c_code": best_c_code,
        "error": last_error,
    }

# 静态文件服务
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

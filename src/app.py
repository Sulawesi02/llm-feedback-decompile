import uvicorn
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List

from config import BASE_MODEL_DIR_PATH, MERGED_MODEL_DIR_PATH
from utils import (
    ModelRunner,
    construct_decompile_prompt,
    compile_to_object,
    build_test_harness,
    run_tests_with_harness,
    construct_refine_decompile_prompt,
)

app = FastAPI()

MODEL_PATH = str(MERGED_MODEL_DIR_PATH / "Qwen2.5-Coder-7B-Instruct" / "v1")
MAX_ITERS = 3 # 最大迭代次数

model_runner = None

class DecompileRequest(BaseModel):
    arch: str # x86 / arm
    opt: str # -O0 / -O1 / -O2 / -O3
    machine_code: str
    test_cases: Optional[List[dict]] = None

@app.on_event("startup")
async def startup_event():
    global model_runner
    model_runner = ModelRunner(MODEL_PATH)

@app.post("/feedback_decompile")
async def feedback_decompile(request: DecompileRequest):
    if not request.machine_code.strip():
        raise HTTPException(status_code=400, detail="机器码不能为空")

    machine_code = request.machine_code.strip() # 机器码
    arch = request.arch # 架构
    opt = request.opt # 优化级别
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
        print("生成为空")
        return {"final_c_code": "// 模型服务调用失败"}

    for it in range(MAX_ITERS):
        print(f"\n--- 迭代 {it+1}/{MAX_ITERS} ---")
        compile_result = compile_to_object(c_code, arch, opt)
        workdir = compile_result.get("workdir")

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
    return {"final_c_code": final_c_code}

# 静态文件服务
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

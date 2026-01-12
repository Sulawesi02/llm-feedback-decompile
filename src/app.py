import uvicorn
import shutil
import requests
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List

from utils import construct_initial_prompt, construct_refine_prompt, clean_c_code, compile_c_code

app = FastAPI()

MODEL_SERVER_URL = "http://localhost:8001/generate"

# 请求模型
class DecompileRequest(BaseModel):
    arch: str # x86 / arm
    opt: str # -O0 / -O1 / -O2 / -O3
    machine_code: str

def generate_c_code(messages: list) -> str:
    """ 调用模型服务生成 C 代码 """
    try:
        print(f"调用模型服务请求: {messages}")
        response = requests.post(MODEL_SERVER_URL, json=messages, timeout=120)
        response.raise_for_status()
        result = response.json()
        c_code = result.get("text", "")
        return clean_c_code(c_code)
    except Exception as e:
        print(f"调用模型服务失败: {e}")
        return ""

# 单次反编译
@app.post("/decompile")
async def decompile(request: DecompileRequest):
    if not request.machine_code.strip():
        raise HTTPException(status_code=400, detail="机器码不能为空")

    prompt = construct_initial_prompt(request.arch, request.opt, request.machine_code)
    c_code = generate_c_code(prompt)
    return {"c_code": c_code or "// 生成失败，可能是模型服务不可用"}

# 循环反馈反编译
@app.post("/feedback_decompile")
async def feedback_decompile(request: DecompileRequest, max_iters: int = 10):
    if not request.machine_code.strip():
        raise HTTPException(status_code=400, detail="机器码不能为空")

    machine_code = request.machine_code.strip()
    arch = request.arch
    opt = request.opt

    history = []
    best_c_code = None

    prompt = construct_initial_prompt(arch, opt, machine_code)
    c_code = generate_c_code(prompt)
    history.append({"iter": 0, "c_code": c_code, "status": "generated"})

    if not c_code:
        return {
            "final_c_code": "// 模型服务调用失败",
            "success": False,
            "total_iterations": 0,
            "history": history
        }

    for it in range(max_iters):
        compile_result = compile_c_code(c_code, arch, opt)
        workdir = compile_result.get("workdir")

        if compile_result["success"]:
            best_c_code = c_code
            history.append({
                "iter": it,
                "c_code": c_code,
                "status": "compile_success",
                "message": "编译成功！"
            })
            # 清理临时目录
            if workdir and Path(workdir).exists():
                shutil.rmtree(workdir, ignore_errors=True)
            break
        else:
            error_msg = compile_result["error"][:1000]  # 截断过长错误
            history.append({
                "iter": it,
                "c_code": c_code,
                "status": "compile_failed",
                "error": error_msg
            })

            refine_prompt = construct_refine_prompt(c_code, error_msg)
            c_code = generate_c_code(refine_prompt)
            if not c_code:
                 history.append({"iter": it + 1, "c_code": "", "status": "error", "message": "模型生成失败"})
                 # 清理临时目录
                 if workdir and Path(workdir).exists():
                    shutil.rmtree(workdir, ignore_errors=True)
                 break

        # 清理临时目录（无论成功失败）
        if workdir and Path(workdir).exists():
            shutil.rmtree(workdir, ignore_errors=True)

    # 最终返回
    return {
        "final_c_code": best_c_code or c_code,
        "success": best_c_code is not None,
        "total_iterations": len(history),
        "history": history
    }

# 静态文件服务
STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir()

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

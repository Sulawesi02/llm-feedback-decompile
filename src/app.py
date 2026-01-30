import uvicorn
import time
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager

from config import (
    MODEL_NAME,
    BASE_MODEL_DIR, 
    SFT_ADAPTER_DIR,
    DPO_ADAPTER_DIR,
    VERSIONS, 
    OFFLOAD_ROOT,
)
from utils import (
    ModelRunning,    
    machine_code_to_binary,
    disasm_binary,
    extract_asm,
    clean_code_block,
    compile_to_object,
)
from src.prompts import construct_infer_prompt, construct_fix_prompt

VERSION = VERSIONS[2][0] # 版本号

MAX_ITERS = 3 # 最大迭代次数

model_runner = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    sft_adapter_path = SFT_ADAPTER_DIR / VERSION
    dpo_adapter_path = DPO_ADAPTER_DIR / VERSION

    global model_runner
    
    offload_dir = OFFLOAD_ROOT / "app"
    offload_dir.mkdir(parents=True, exist_ok=True)
    
    model_runner = ModelRunning(
        base_model_path=str(base_model_path),
        sft_adapter_path=str(sft_adapter_path),
        dpo_adapter_path=str(dpo_adapter_path),
        offload_folder=str(offload_dir),
        offload_buffers=False
    )
    try:
        yield
    finally:
        if model_runner:
            model_runner.unload()

app = FastAPI(lifespan=lifespan)

class DecompileRequest(BaseModel):
    arch: str # x86 / arm
    machine_code: str

@app.post("/decompile")
async def decompile(request: DecompileRequest):
    if not request.machine_code.strip():
        raise HTTPException(status_code=400, detail="机器码不能为空")

    arch = request.arch
    machine_code = request.machine_code

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
                print("编译成功。")
                best_c_code = generate_c_code
                history.append({
                    "iter": it,
                    "success": True,
                    "generate_c_code": generate_c_code,
                    "error": "",
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
    return {"best_c_code": best_c_code} 

# 静态文件服务
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

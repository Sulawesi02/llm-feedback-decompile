import uvicorn
import torch
import shutil
import gc
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    MODEL_NAME,
    BASE_MODEL_DIR, 
    DPO_ADAPTER_DIR,
    VERSIONS, 
    QUANT_CONFIG,
)
from utils import ( 
    machine_code_to_binary,
    disasm_binary,
    extract_asm,
    compile_to_object,
)
from src.prompts import construct_infer_prompt, construct_fix_prompt

VERSION = VERSIONS[2][0] # 版本号

MAX_ITERS = 3 # 最大迭代次数

tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    base_model_path = BASE_MODEL_DIR / MODEL_NAME
    dpo_adapter_path = DPO_ADAPTER_DIR / VERSION

    global tokenizer, model
    
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    
    print("加载模型...")
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            quantization_config=QUANT_CONFIG,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ),
        str(dpo_adapter_path),
        device_map={"": "cuda:0"},
    )
    model.eval() # 推理模式
    try:
        yield
    finally:
        # 清理资源
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

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
            
            inputs = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            print("生成 C 函数代码...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            outputs = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1]

            print("编译 C 函数代码...")
            success, error_msg, o_path = compile_to_object(arch, outputs)
            if success:
                print("编译成功")
                best_outputs = outputs
                history.append({
                    "iter": it,
                    "success": True,
                    "outputs": outputs,
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
    return {"best_outputs": best_outputs} 

# 静态文件服务
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
